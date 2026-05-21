"""Wiki Content Writer — per-page writer agent (#285, #314).

This module provides:

``compute_page_budget(target_symbols, target_folders, target_docs, base)``
    Pure function; derives the tool-call budget for a single page from its
    spec.  Used by ``WikiContentWriter.compute_budget``.

``WikiContentWriter``
    Class wired to a ``WriterTools`` instance.  Runs the per-page agentic
    write loop: budget calculation → tool-augmented LLM call → markdown.

Budget formula (#314)
---------------------
budget = base
       + 2   * count(target_symbols where layer in {entry_point, public_api})
       + 1.5 * count(target_symbols where layer in {core_type, infrastructure})
       + 1   * count(target_folders)
       + 1   * count(target_docs)

Soft target: 20-40 calls.  Hard cap: 80 calls.  Default base: 8.

Since #309 landed LLM summaries in evidence packs the writer uses tool calls
primarily for line-number precision rather than symbol discovery, so budgets
can be higher without proportionally more cost.
"""

from __future__ import annotations

import asyncio
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
    from app.core.wiki_structure_planner.evidence import EvidenceSlice
    from app.core.wiki_structure_planner.planner_agent import ChapterSpec, SubPageSpec

logger = logging.getLogger(__name__)

# ── Budget constants ──────────────────────────────────────────────────────────

BUDGET_SOFT_MIN: int = 20   # #314: raised from 15
BUDGET_SOFT_MAX: int = 40   # #314: raised from 30
BUDGET_HARD_CAP: int = 80

# Default base budget — reads LLM_WRITER_BASE_BUDGET from Settings so the value
# is consistent with the rest of the app config. Falls back to 8 when Settings
# is unavailable (test isolation, import order issues).
try:
    from app.config import get_settings as _get_settings  # noqa: PLC0415
    _DEFAULT_BASE_BUDGET: int = _get_settings().llm_writer_base_budget
except Exception:
    _DEFAULT_BASE_BUDGET = 8

# Symbol layers weighted at 2× (high-importance surface; entry_point / public_api)
_WEIGHT_2_LAYERS: frozenset[str] = frozenset({"entry_point", "public_api"})

# Symbol layers weighted at 1.5× (#314: raised from 1; supporting architectural types)
_WEIGHT_1_5_LAYERS: frozenset[str] = frozenset({"core_type", "infrastructure"})


# ── Budget formula ────────────────────────────────────────────────────────────


def compute_page_budget(
    target_symbols: list[tuple[str, str]],
    target_folders: list[str],
    target_docs: list[str],
    base: int = _DEFAULT_BASE_BUDGET,
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
        Baseline budget (default 8, #314) before any scaling.  Override via
        ``LLM_WRITER_BASE_BUDGET`` env var or the ``base`` field in a PageSpec.

    Returns
    -------
    int
        Tool-call budget, clamped to ``BUDGET_HARD_CAP``.
    """
    weight_2 = sum(1 for _, layer in target_symbols if layer in _WEIGHT_2_LAYERS)
    weight_1_5 = sum(1 for _, layer in target_symbols if layer in _WEIGHT_1_5_LAYERS)

    budget = base + 2 * weight_2 + int(1.5 * weight_1_5) + len(target_folders) + len(target_docs)
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

### Why citations matter

The citation verifier checks every paragraph against the actual source code.
A paragraph with no citation is treated as ungrounded prose and is discarded
entirely — it will not appear in the final wiki page. A citation pointing to a
line range that does not exist in the repository is also discarded. Only
paragraphs with citations that match real, readable code survive into the
final output. This means: if you write a sentence without a citation, it will
be silently removed. If you cite a line range that you did not actually read
with read_file, the verifier may reject it. The only safe approach is to call
read_file, observe the line numbers in the returned content, and then cite
those exact lines. Never guess a line number. Never cite a file you did not
open with read_file or confirm with get_signature.

### Citation rule details

Rule 1 — Every claim needs a citation. The consequence of omitting a citation
is that the whole paragraph is stripped. Write claims in the form:
"The scheduler uses a one-second poll interval [src/scheduler.py:88]."
Append the citation token immediately after the claim, before the sentence's
closing punctuation. Do not move it to the end of the paragraph if the
paragraph covers multiple files — cite each claim individually.

Rule 2 — README content must be verified. If you intend to reference something
described in a README, you must call read_file on the README first and cite
the specific lines. Paraphrasing README text without a matching citation is
treated as an uncited claim and will be stripped.

Rule 3 — Identifiers must appear in your tool trace. Do not name a function,
class, or environment variable that did not appear in the output of read_file,
get_signature, get_callers, get_callees, grep, or list_doc_chunks. If you are
uncertain whether a symbol exists, call get_signature or grep before writing
about it. Mentioning a non-existent symbol causes the paragraph to be flagged
and may cause the entire page to fail verification.

## Available tools

**read_file(path, start_line?, end_line?)** — Read lines from a repo-relative file.
Returns: numbered lines in the form `<N>: <line_content>` (one per line).
  Example: `47: class JobQueue:\\n48:     def __init__(self, max_size: int = 1000):`
Use: read files containing the symbols in target_symbols. The line numbers
in the output are the exact numbers to use in citations: read line 47 → cite
[path:47] or [path:47-48] for a range.
Error: `[error] ...` — try a different path or call get_signature first.

**get_signature(symbol)** — Look up a symbol's definition location and signature.
Returns: `file_path: signature (layer)\\ndocstring`
  Example: `src/workers/queue.py: class JobQueue (infrastructure)\\nAsyncIO job queue.`
Use: confirm a class or function exists and find its file path before calling
read_file. Does NOT return line numbers — use read_file after get_signature
to get the exact lines for citations.
Not found: `[not found] symbol` — do not mention this symbol in the page.

**grep(pattern)** — Full-text search across source files.
Returns: `file_path:line_number: line_text` (up to 20 matches)
Use: find where a concept or identifier is used across the codebase. Use grep
to locate configuration values, string literals, or identifiers you know exist
but whose file path you are unsure of.
No matches: `[no matches]` — the pattern does not appear in source files.

**get_callers(symbol)** — List symbols that call the given symbol.
Returns: `file_path: symbol_name` per caller
Use: document who uses a function or class (integration context). Useful for
writing the "used by" section of a public API page. Shows the call graph from
the perspective of dependents.

**get_callees(symbol)** — List symbols the given symbol calls.
Returns: `file_path: symbol_name` per callee
Use: document what a function or class depends on (dependency chain). Useful
for showing what a component orchestrates or delegates to.

**list_doc_chunks(doc_path)** — Return documentation sections for a doc file.
Returns: `[N] heading\\ntext` per chunk
Use: incorporate doc context into pages that cover documented modules. Prefer
this over read_file when you need the logical structure of a markdown document
rather than raw line content.

## Citation examples

Good — claim immediately followed by a bracketed citation token:
  "The worker polls with a 1-second timeout [src/workers/worker.py:62-63]."
  "Default queue size is 1000 items [src/workers/queue.py:57]."
  "Startup recovery loads pending jobs [src/workers/queue.py:121-133]."
  "Config is read from environment [config.py:5-12] and merged with file
  defaults [config.py:40-48]."

Bad — NEVER write these (they are discarded by the verifier):
  "The worker polls with a 1-second timeout." ← no citation, discarded
  "See worker.py for the timeout logic." ← path in prose not in brackets, discarded
  "The worker polls (worker.py:62)." ← parens not brackets, citation not parsed
  "The queue has configurable size [queue.py:57]." ← missing src/ prefix,
    may not resolve to a real path — always use the exact path from read_file
  "The scheduler module handles retries." ← whole paragraph with no citation,
    discarded in full regardless of how accurate the claim is

## Budget guidance

You have a finite number of tool calls available for this page. Use them to
read the files and symbols listed in the page spec before writing. Prioritise
reading symbols from target_symbols first, then explore related callers or
callees if the budget allows. Once you have read enough to ground every claim,
write the complete page. Do not call tools after you have started writing.

IMPORTANT: every substantive paragraph MUST end with at least one citation in
the format shown above. A paragraph with no citation will be stripped by the
verifier. Start by calling tools to read the relevant files, then write the
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
        - ``base``: baseline budget (optional, default 8 per #314).

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
            base=int(page_spec.get("base", _DEFAULT_BASE_BUDGET)),
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
                "base": getattr(page_spec, "base", _DEFAULT_BASE_BUDGET),
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

    # ── generate_chapter ──────────────────────────────────────────────────

    async def generate_chapter(
        self,
        chapter_spec: "ChapterSpec",
        evidence_slices: "list[EvidenceSlice]",
        *,
        stream_callback: Callable[[dict], None] | None = None,
    ) -> tuple[str, list[str]]:
        """Generate a chapter: N sub-pages in parallel + one chapter index.

        For each ``(subpage, evidence_slice)`` pair in
        ``zip(chapter_spec.subpages, evidence_slices)``, builds a
        page-spec dict and calls ``self.generate_page`` in parallel via
        ``asyncio.gather``.  After all sub-pages are done, generates the
        chapter index: a short LLM-written overview paragraph followed by a
        ``[[wikilink]]`` TOC.

        Parameters
        ----------
        chapter_spec :
            Chapter produced by ``run_planner_chaptered``.
        evidence_slices :
            One ``EvidenceSlice`` per subpage (zipped by position).
        stream_callback :
            Optional SSE event emitter.  Receives ``writer.tool_call`` and
            ``writer.subpage_oversized`` events in addition to all per-page
            events from ``generate_page``.

        Returns
        -------
        (chapter_index_markdown, list_of_subpage_markdowns)
            ``chapter_index_markdown`` — short overview + ``[[wikilink]]`` TOC.
            ``list_of_subpage_markdowns`` — one entry per sub-page (never empty
            when subpages exist; fallback applied on per-subpage error).
        """
        loop = asyncio.get_running_loop()

        subpages = chapter_spec.subpages

        async def _generate_one(sp: "SubPageSpec", sl: "EvidenceSlice") -> str:
            """Generate one sub-page; return fallback markdown on any error."""
            page_spec_dict = {
                "title": sp.title,
                "description": sp.description,
                "retrieval_query": sp.retrieval_query,
                "target_symbols": sl.symbols,
                "target_folders": sl.file_paths,
                "target_docs": sl.doc_paths,
                "cluster_id": chapter_spec.cluster_id,
            }
            try:
                md, _ = await loop.run_in_executor(
                    None,
                    lambda: self.generate_page(page_spec_dict, stream_callback=stream_callback),
                )
            except Exception as exc:
                logger.warning(
                    "[WRITER] generate_chapter sub-page failed: title=%r error=%s",
                    sp.title,
                    exc,
                )
                md = f"# {sp.title}\n\n{sp.description}\n"

            n_lines = md.count("\n")
            if n_lines > 250:
                logger.info(
                    "[WRITER] sub-page oversized: title=%r lines=%d",
                    sp.title,
                    n_lines,
                )
                if stream_callback:
                    try:
                        stream_callback({
                            "event": "writer.subpage_oversized",
                            "title": sp.title,
                            "lines": n_lines,
                            "cluster_id": chapter_spec.cluster_id,
                        })
                    except Exception:
                        pass

            return md

        # ── 1. Generate all sub-pages in parallel ─────────────────────
        tasks = [
            _generate_one(sp, sl)
            for sp, sl in zip(subpages, evidence_slices)
        ]
        subpage_mds: list[str] = list(await asyncio.gather(*tasks))

        # ── 2. Build chapter index via a direct LLM call ──────────────
        # LLM writes ONLY the 2-3 sentence overview — no TOC in the prompt.
        # The static [[wikilink]] TOC is appended deterministically after,
        # so we never ask the LLM to reproduce it (which would double it).
        index_prompt = (
            f'Write a brief 2-3 sentence overview of the chapter '
            f'"{chapter_spec.chapter_title}": {chapter_spec.chapter_description}. '
            f"Focus on what the reader will learn and the key components covered. "
            f"Output only the overview paragraph — do not list sub-pages."
        )

        effective_llm = self.llm_client
        index_overview = ""
        if effective_llm is not None:
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: effective_llm.invoke([HumanMessage(content=index_prompt)]),
                )
                index_overview = (
                    response.content
                    if isinstance(response.content, str)
                    else str(response.content)
                )
            except Exception as exc:
                logger.warning("[WRITER] chapter index LLM call failed: %s", exc)

        # ── 3. Append the [[wikilink]] TOC deterministically ─────────
        toc_wikilinks = "\n".join(
            f"- [[{sp.title}]]: {sp.description}"
            for sp in subpages
        )
        chapter_index_md = f"{index_overview}\n\n{toc_wikilinks}".strip()

        logger.info(
            "[WRITER] generate_chapter done: title=%r subpages=%d index_chars=%d",
            chapter_spec.chapter_title,
            len(subpage_mds),
            len(chapter_index_md),
        )
        return chapter_index_md, subpage_mds

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
