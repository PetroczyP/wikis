"""Wiki Content Writer — per-page writer agent (#285).

This module provides:

``compute_page_budget(target_symbols, target_folders, target_docs, base)``
    Pure function; derives the tool-call budget for a single page from its
    spec.  Used by ``WikiContentWriter.compute_budget``.

``WikiContentWriter``
    Class wired to a ``WriterTools`` instance.  Runs the per-page agentic
    write loop: budget calculation → tool-augmented LLM call → markdown.

Budget formula
--------------
budget = base
       + 2 * count(target_symbols where layer in {entry_point, public_api})
       + 1 * count(target_symbols where layer in {core_type, infrastructure})
       + 1 * count(target_folders)
       + 1 * count(target_docs)

Soft target: 15-30 calls.  Hard cap: 80 calls.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.core.agent_tools import WIKI_TOOL_SCHEMAS as _WRITER_TOOL_SCHEMAS
from app.services.llm_factory import is_bedrock_llm, make_cached_system_message, safe_bind_tools

from .prompts import CITATION_FORMAT_DESCRIPTION, CITATION_PROMPT_RULES
from .source_gate import ToolCall
from .writer_tools import WriterTools

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ── Budget constants ──────────────────────────────────────────────────────────

BUDGET_SOFT_MIN: int = 15
BUDGET_SOFT_MAX: int = 30
BUDGET_HARD_CAP: int = 80

# Symbol layers weighted at 2× (high-importance surface)
_WEIGHT_2_LAYERS: frozenset[str] = frozenset({"entry_point", "public_api"})

# Symbol layers weighted at 1× (supporting types)
_WEIGHT_1_LAYERS: frozenset[str] = frozenset({"core_type", "infrastructure"})


# ── Budget formula ────────────────────────────────────────────────────────────


def compute_page_budget(
    target_symbols: list[tuple[str, str]],
    target_folders: list[str],
    target_docs: list[str],
    base: int = 5,
) -> int:
    """Compute the tool-call budget for a single wiki page.

    Parameters
    ----------
    target_symbols : list of (name, layer) tuples
        Symbols the page should document.  Layer is one of
        ``entry_point | public_api | core_type | infrastructure`` (others
        contribute 0).
    target_folders : list[str]
        Repository folders the page covers.
    target_docs : list[str]
        Documentation files referenced by the page.
    base : int
        Baseline budget (default 5) before any scaling.

    Returns
    -------
    int
        Tool-call budget, clamped to ``BUDGET_HARD_CAP``.
    """
    weight_2 = sum(1 for _, layer in target_symbols if layer in _WEIGHT_2_LAYERS)
    weight_1 = sum(1 for _, layer in target_symbols if layer in _WEIGHT_1_LAYERS)

    budget = base + 2 * weight_2 + 1 * weight_1 + len(target_folders) + len(target_docs)
    return min(budget, BUDGET_HARD_CAP)


# ── WikiContentWriter ─────────────────────────────────────────────────────────


# Writer tool schemas — imported from the shared catalog at top of file.
# Aliased ``_WRITER_TOOL_SCHEMAS`` for back-compat with callers that imported
# the old private name (and to preserve the local-symbol narrative in tests
# that introspect ``llm.bind_tools_schemas``).

# System prompt prefix — matches the role-detection prefix in integration tests.
_SYSTEM_PROMPT_PREFIX = "You are a technical documentation writer."

_SYSTEM_PROMPT_TEMPLATE = """{prefix}

Your job is to write a detailed, accurate wiki page in Markdown based on the
page specification and your tool results.

## Citation contract

{format_description}

Rules:
{rules}

## Worked examples

A claim ending with a single-line citation:
  "The token is validated in `verify_token` [api/auth.py:42]."

A claim ending with a range citation:
  "Session expiry logic lives in the middleware [middleware/session.py:10-30]."

A paragraph citing two files:
  "Config is loaded from environment [config.py:5-12] and merged with defaults
  [config.py:40-48]."

IMPORTANT: every substantive paragraph MUST end with at least one citation in
the format shown above. A paragraph with no citation will be stripped by the
verifier.  Start by calling tools to read the relevant files, then write the
page with inline citations after every claim.
"""

_USER_PROMPT_TEMPLATE = """\
## Page to write

**Title:** {title}
**Description:** {description}
**Retrieval query:** {retrieval_query}

**Target symbols:** {symbols}
**Target folders:** {folders}
**Target docs:** {docs}

First, read the most relevant files from target symbols / folders using the
available tools.  Then write the complete wiki page in Markdown with inline
`[path:N]` or `[path:lo-hi]` citations after every claim.
"""


class WikiContentWriter:
    """Per-page writer agent for wiki content generation.

    Runs a manual tool-augmented LLM loop: the LLM may call ``read_file``,
    ``get_signature``, ``grep``, etc. to explore the repository, then writes
    the final page with inline citation tokens.

    Parameters
    ----------
    tools : WriterTools
        Tool surface the agent uses to explore the repository.
    llm_client : BaseChatModel | None
        Pre-configured LangChain chat model.
    max_iterations : int
        Hard iteration ceiling.  Defaults to ``BUDGET_HARD_CAP``.
    """

    def __init__(
        self,
        tools: WriterTools,
        llm_client: Any | None = None,
        max_iterations: int = BUDGET_HARD_CAP,
    ) -> None:
        self.tools = tools
        self.llm_client = llm_client
        self.max_iterations = max_iterations

    # ── Budget ────────────────────────────────────────────────────────────

    def compute_budget(self, page_spec: dict[str, Any]) -> int:
        """Derive the tool-call budget from a page spec dict.

        Accepts the same key structure as ``PageSpec`` (from
        ``app.core.state.wiki_state``):

        - ``target_symbols``: list of (name, layer) tuples or list of str
          (bare names; layer defaults to ``""``).
        - ``target_folders``: list of folder paths.
        - ``target_docs``: list of doc file paths.
        - ``base``: baseline budget (optional, default 5).

        Returns
        -------
        int
            Tool-call budget, capped at ``BUDGET_HARD_CAP``.
        """
        raw_symbols = page_spec.get("target_symbols", [])

        # Normalise: accept both (name, layer) tuples and bare strings
        symbols: list[tuple[str, str]] = []
        for entry in raw_symbols:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                symbols.append((str(entry[0]), str(entry[1])))
            else:
                # Bare symbol name — layer unknown, contributes 0
                symbols.append((str(entry), ""))

        return compute_page_budget(
            target_symbols=symbols,
            target_folders=list(page_spec.get("target_folders", [])),
            target_docs=list(page_spec.get("target_docs", [])),
            base=int(page_spec.get("base", 5)),
        )

    # ── generate_page ─────────────────────────────────────────────────────

    def generate_page(
        self,
        page_spec: Any,  # PageSpec dataclass or dict — both supported
        *,
        llm: Any | None = None,
        stream_callback: Callable[[dict], None] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """Generate markdown for one wiki page with tool-augmented LLM calls.

        Runs a bounded tool-call loop: the LLM may invoke ``read_file``,
        ``get_signature``, ``grep``, etc. to ground its claims, then writes
        the page with inline ``[path:N]`` citations.

        Parameters
        ----------
        page_spec:
            Either a ``PageSpec`` dataclass (from the planner) or a plain dict
            with the same fields.  Required keys: ``title``, ``description``,
            ``retrieval_query``, ``target_symbols``, ``target_folders``,
            ``target_docs``.
        llm:
            Override LLM for this call.  Falls back to ``self.llm_client``.
        stream_callback:
            Optional event emitter.  Receives a dict per tool call.

        Returns
        -------
        (markdown, tool_trace)
            ``markdown`` — final page content (may contain citation tokens).
            ``tool_trace`` — list of ``ToolCall`` objects from the agent loop.
        """
        effective_llm = llm if llm is not None else self.llm_client
        if effective_llm is None:
            raise ValueError("generate_page: no LLM configured (pass llm= or set llm_client)")

        # Normalise spec to dict-like access
        if isinstance(page_spec, dict):
            spec = page_spec
        else:
            # PageSpec dataclass
            spec = {
                "title": getattr(page_spec, "title", ""),
                "description": getattr(page_spec, "description", ""),
                "retrieval_query": getattr(page_spec, "retrieval_query", ""),
                "target_symbols": getattr(page_spec, "target_symbols", []),
                "target_folders": getattr(page_spec, "target_folders", []),
                "target_docs": getattr(page_spec, "target_docs", []),
            }

        title = spec.get("title", "")
        description = spec.get("description", "")
        retrieval_query = spec.get("retrieval_query", "")
        target_symbols = spec.get("target_symbols", [])
        target_folders = spec.get("target_folders", [])
        target_docs = spec.get("target_docs", [])

        budget = self.compute_budget(spec)
        logger.info("[WRITER] generate_page: title=%r budget=%d", title, budget)

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            prefix=_SYSTEM_PROMPT_PREFIX,
            format_description=CITATION_FORMAT_DESCRIPTION,
            rules="\n".join(f"{i + 1}. {r}" for i, r in enumerate(CITATION_PROMPT_RULES)),
        )
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            title=title,
            description=description,
            retrieval_query=retrieval_query,
            symbols=", ".join(str(s) for s in target_symbols) or "none",
            folders=", ".join(str(f) for f in target_folders) or "none",
            docs=", ".join(str(d) for d in target_docs) or "none",
        )

        # Use safe_bind_tools to unwrap any retry/fallback wrappers before binding.
        # wiki_runner passes the LLM without tools=, so the factory's with_retry()
        # wrap sits outermost; RunnableRetry has no bind_tools — calling it directly
        # raises AttributeError and silently disables tools (#295).
        llm_with_tools = safe_bind_tools(effective_llm, _WRITER_TOOL_SCHEMAS)

        messages: list[Any] = [
            make_cached_system_message(system_prompt, cache=is_bedrock_llm(effective_llm)),
            HumanMessage(content=user_prompt),
        ]

        tool_trace: list[ToolCall] = []
        # Cap is expressed in total tool invocations, matching the docstring's
        # "tool-call budget" language.  A single LLM response may contain N
        # tool calls, so we track the cumulative count — not the round count —
        # to ensure we never exceed BUDGET_HARD_CAP.
        tool_call_budget = min(budget, self.max_iterations)
        tool_calls_made = 0
        content = ""

        for _round in range(tool_call_budget):
            response = llm_with_tools.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # LLM wrote the page — done.
                content = response.content if isinstance(response.content, str) else str(response.content)
                break

            # Execute ALL tool calls in this response before checking budget.
            # Breaking mid-response leaves tool_use blocks without corresponding
            # tool_result blocks — Bedrock rejects this with ValidationException.
            messages.append(response)
            for tc in tool_calls:
                tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tc_id = tc.get("id", str(_round)) if isinstance(tc, dict) else getattr(tc, "id", str(_round))

                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except json.JSONDecodeError:
                        tc_args = {}
                if not isinstance(tc_args, dict):
                    tc_args = {}

                result_text = self._dispatch_tool(tc_name, tc_args)
                tool_calls_made += 1
                tool_trace.append(ToolCall(tool=tc_name, args=tc_args, result_text=result_text))

                if stream_callback:
                    try:
                        stream_callback({"event": "writer.tool_call", "tool": tc_name, "args": tc_args})
                    except Exception:
                        pass

                messages.append(ToolMessage(content=result_text, tool_call_id=tc_id, name=tc_name))

            logger.debug("[WRITER] tool round %d/%d: %d calls this round, %d total",
                         _round + 1, tool_call_budget, len(tool_calls), tool_calls_made)

            # Check budget AFTER finishing the full response (never mid-response).
            if tool_calls_made >= tool_call_budget:
                break

        if not content:
            # Budget exhausted without a final write — explicitly instruct the
            # LLM to write the page and cite every claim using the file paths
            # and line numbers already in the message history (#313).
            messages.append(HumanMessage(content=(
                "You have used all available tool calls. Based on the evidence "
                "you gathered, write the complete wiki page in Markdown now.\n\n"
                "IMPORTANT: cite every claim using the file path and line numbers "
                "from your tool results. Format: [path/to/file.py:N] or "
                "[path/to/file.py:lo-hi]. A paragraph without a citation will be "
                "discarded. Do NOT call any more tools."
            )))
            try:
                response = effective_llm.invoke(messages)
                final_content = response.content if isinstance(response.content, str) else str(response.content)
                # If LLM still returned tool_calls (stubborn), ignore and fall back.
                final_tool_calls = getattr(response, "tool_calls", None) or []
                if final_content and not final_tool_calls:
                    content = final_content
            except Exception as exc:
                logger.error("[WRITER] final write call failed: %s", exc)
            if not content:
                content = f"# {title}\n\n{description}\n"

        logger.info("[WRITER] generate_page done: title=%r chars=%d tool_calls=%d", title, len(content), len(tool_trace))
        return content, tool_trace

    # ── Tool dispatch ─────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        """Route a tool call to the appropriate WriterTools method."""
        try:
            if name == "read_file":
                path = args.get("path", "")
                start = args.get("start_line")
                end = args.get("end_line")
                line_range = (int(start), int(end)) if start is not None and end is not None else None
                result = self.tools.read_file(path, line_range=line_range)
                if result.error:
                    return f"[error] {result.error}"
                lines = result.lines
                numbered = "\n".join(f"{i + (line_range[0] if line_range else 1)}: {ln}" for i, ln in enumerate(lines))
                return numbered or "[empty file]"

            if name == "get_signature":
                sig = self.tools.get_signature(args.get("symbol", ""))
                if not sig.found:
                    return f"[not found] {sig.symbol}"
                return f"{sig.file_path}: {sig.signature} ({sig.layer})\n{sig.docstring}"

            if name == "grep":
                matches = self.tools.grep(args.get("pattern", ""))
                if not matches:
                    return "[no matches]"
                return "\n".join(f"{m.file_path}:{m.line_number}: {m.line_text}" for m in matches[:20])

            if name == "get_callers":
                result = self.tools.get_callers(args.get("symbol", ""))
                if not result.callers:
                    return "[no callers]" if result.found else "[not found]"
                return "\n".join(f"{c.file_path}: {c.symbol}" for c in result.callers)

            if name == "get_callees":
                result = self.tools.get_callees(args.get("symbol", ""))
                if not result.callees:
                    return "[no callees]" if result.found else "[not found]"
                return "\n".join(f"{c.file_path}: {c.symbol}" for c in result.callees)

            if name == "list_doc_chunks":
                chunks = self.tools.list_doc_chunks(args.get("doc_path", ""))
                if not chunks:
                    return "[no chunks]"
                return "\n".join(f"[{c.chunk_index}] {c.heading}\n{c.text}" for c in chunks)

        except Exception as exc:
            logger.warning("[WRITER] tool %r failed: %s", name, exc)
            return f"[error] {exc}"

        return f"[error] unknown tool: {name!r}"
