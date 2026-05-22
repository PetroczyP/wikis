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

from app.prompts import load_prompt

from .diagram_generator import DiagramGenerator
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

_SYSTEM_PROMPT_TEMPLATE = load_prompt("writer/system.md")
_USER_PROMPT_TEMPLATE = load_prompt("writer/user_template.md")


def _extract_llm_text(response: Any) -> str:
    """Extract plain text from an LLM response regardless of provider format.

    Anthropic/Bedrock returns ``response.content`` as a list of content blocks
    (``[{'type': 'text', 'text': '...'}, {'type': 'tool_use', ...}]``).
    Other providers return a plain string.  This helper normalises both.
    """
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif hasattr(block, "type") and getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(parts)
    return str(content)


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
            messages.append(HumanMessage(content=load_prompt("writer/budget_exhausted.md")))
            try:
                response = effective_llm.invoke(messages)
                final_content = _extract_llm_text(response)
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

    # Budget for the chapter-index mini research loop (6 tool calls).
    _CHAPTER_INDEX_BUDGET: int = 6

    @staticmethod
    def _extract_subpage_outline(title: str, md: str, char_cap: int = 3000) -> str:
        """Return a compact heading outline of a generated sub-page.

        Extracts ## and ### headings from *md*, prepended with the page title,
        and truncates at *char_cap* characters.  Used to give the chapter-index
        LLM call context about what was actually written without sending the
        full page.
        """
        headings: list[str] = []
        for line in md.splitlines():
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("### "):
                headings.append(stripped)
        heading_text = "\n".join(headings[:40]) or "(no headings)"
        raw = f"### {title}\n{heading_text}"
        return raw[:char_cap]

    def _generate_chapter_index(
        self,
        chapter_spec: "ChapterSpec",
        evidence_slices: "list[EvidenceSlice]",
        toc_wikilinks: str,
        *,
        subpage_mds: "list[str] | None" = None,
        stream_callback: Callable[[dict], None] | None = None,
    ) -> str:
        """Run a mini tool-augmented loop to write the chapter _index.md.

        Uses up to ``_CHAPTER_INDEX_BUDGET`` (6) tool calls focused on the
        cluster's own symbols to produce a structured Markdown overview with:

        ``## Overview`` — 2-3 sentences describing the cluster.
        ``## Key Components`` — list of main symbols with 1-sentence roles.
        ``## Sub-pages`` — ``[[wikilink]]`` TOC built deterministically.

        When *subpage_mds* is provided, a heading outline of each already-
        generated sub-page is included in the prompt so the index accurately
        reflects the actual content rather than re-reading the codebase from
        scratch.

        Falls back to a minimal stub if the LLM call fails or the client is
        unavailable.  Never raises.
        """
        effective_llm = self.llm_client
        if effective_llm is None:
            return self._chapter_index_fallback(
                chapter_spec.chapter_title, toc_wikilinks
            )

        # Gather seed symbols and file paths from the first evidence slice.
        seed_symbols: list[str] = []
        seed_files: list[str] = []
        for sl in evidence_slices[:2]:  # limit context to first 2 slices
            seed_symbols.extend(sl.symbols[:4])
            seed_files.extend(sl.file_paths[:3])

        # Build outlines from already-generated sub-pages so the LLM index
        # reflects the actual written content instead of re-deriving it.
        subpage_context = ""
        if subpage_mds and chapter_spec.subpages:
            outlines = [
                self._extract_subpage_outline(sp.title, md)
                for sp, md in zip(chapter_spec.subpages, subpage_mds)
            ]
            subpage_context = (
                "\n\nAlready-generated sub-page outlines (use these to write "
                "an accurate index — do NOT contradict what was written):\n"
                + "\n\n".join(outlines)
            )

        # Build the wikilink TOC line so the LLM can reference it in the
        # ## Sub-pages section (the deterministic version is appended after).
        system_prompt = (
            f"You are writing the overview page for a wiki chapter about "
            f"{chapter_spec.chapter_title}.\n"
            "Your goal: write a clear, human-readable _index.md that helps "
            "developers understand this cluster of code. Use tools to read "
            "2-3 key files and understand what the code does.\n\n"
            "Output ONLY this Markdown structure (no extra commentary):\n"
            "## Overview\n"
            "[2-3 sentences explaining what this cluster does and why it "
            "exists in the system]\n\n"
            "## Key Components\n"
            "- **SymbolName**: [1 sentence role]\n"
            "- **SymbolName**: [1 sentence role]\n"
            "...\n\n"
            "## Sub-pages\n"
            f"{toc_wikilinks}\n\n"
            "IMPORTANT: Do NOT write code. Write only prose descriptions. "
            "Do NOT call more than 6 tools total."
        )

        user_prompt = (
            f"Chapter: {chapter_spec.chapter_title}\n"
            f"Description: {chapter_spec.chapter_description}\n\n"
            f"Key symbols: {', '.join(seed_symbols) or 'none'}\n"
            f"Key files: {', '.join(seed_files) or 'none'}\n\n"
            f"Sub-pages to document:\n{toc_wikilinks}"
            f"{subpage_context}\n\n"
            "Read 2-3 key files to understand the code, then write the "
            "chapter index following the system prompt structure exactly."
        )

        llm_with_tools = safe_bind_tools(effective_llm, _WRITER_TOOL_SCHEMAS)

        messages: list[Any] = [
            make_cached_system_message(system_prompt, cache=is_bedrock_llm(effective_llm)),
            HumanMessage(content=user_prompt),
        ]

        tool_calls_made = 0
        content = ""

        try:
            for _round in range(self._CHAPTER_INDEX_BUDGET):
                response = llm_with_tools.invoke(messages)
                tool_calls = getattr(response, "tool_calls", None) or []

                if not tool_calls:
                    content = (
                        response.content
                        if isinstance(response.content, str)
                        else str(response.content)
                    )
                    break

                # Execute all tool calls in this round.
                messages.append(response)
                for tc in tool_calls:
                    tc_name = (
                        tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    )
                    tc_args = (
                        tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    )
                    tc_id = (
                        tc.get("id", str(_round))
                        if isinstance(tc, dict)
                        else getattr(tc, "id", str(_round))
                    )

                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except json.JSONDecodeError:
                            tc_args = {}
                    if not isinstance(tc_args, dict):
                        tc_args = {}

                    result_text = self._dispatch_tool(tc_name, tc_args)
                    tool_calls_made += 1

                    if stream_callback:
                        try:
                            stream_callback({
                                "event": "writer.tool_call",
                                "tool": tc_name,
                                "args": tc_args,
                            })
                        except Exception:
                            pass

                    messages.append(
                        ToolMessage(content=result_text, tool_call_id=tc_id, name=tc_name)
                    )

                logger.debug(
                    "[WRITER] chapter index tool round %d/%d: %d calls this round, %d total",
                    _round + 1,
                    self._CHAPTER_INDEX_BUDGET,
                    len(tool_calls),
                    tool_calls_made,
                )

                if tool_calls_made >= self._CHAPTER_INDEX_BUDGET:
                    # Budget exhausted — force final write.
                    messages.append(HumanMessage(content=load_prompt("writer/chapter_index_budget_exhausted.md")))
                    response = llm_with_tools.invoke(messages)
                    final_tc = getattr(response, "tool_calls", None) or []
                    if not final_tc:
                        content = (
                            response.content
                            if isinstance(response.content, str)
                            else str(response.content)
                        )
                    break

        except Exception as exc:
            logger.warning("[WRITER] chapter index loop failed: %s", exc)

        if not content:
            return self._chapter_index_fallback(chapter_spec.chapter_title, toc_wikilinks)

        # Ensure the deterministic Sub-pages TOC is always present.
        if "## Sub-pages" not in content:
            content = f"{content.rstrip()}\n\n## Sub-pages\n{toc_wikilinks}"
        elif toc_wikilinks and toc_wikilinks.strip() not in content:
            # Replace everything from "## Sub-pages" to end-of-string with the
            # deterministic TOC. Keep everything the LLM wrote BEFORE that section.
            sub_pos = content.find("## Sub-pages")
            content = f"{content[:sub_pos].rstrip()}\n\n## Sub-pages\n{toc_wikilinks}"

        logger.info(
            "[WRITER] chapter index done: title=%r tool_calls=%d chars=%d",
            chapter_spec.chapter_title,
            tool_calls_made,
            len(content),
        )
        return content

    def _chapter_index_fallback(self, chapter_title: str, toc_wikilinks: str) -> str:
        """Minimal chapter index when the LLM loop fails or is unavailable."""
        parts = [
            "## Overview",
            f"{chapter_title}.",
            "",
            "## Key Components",
            "",
        ]
        if toc_wikilinks:
            parts += ["## Sub-pages", toc_wikilinks]
        return "\n".join(parts)

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
        chapter index via a mini tool-augmented loop (budget=6) that reads
        key files from the cluster before writing a structured
        ``## Overview / ## Key Components / ## Sub-pages`` index.

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
            ``chapter_index_markdown`` — structured overview + wikilink TOC.
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

        # ── 2. Build the deterministic [[wikilink]] TOC ───────────────
        toc_wikilinks = "\n".join(
            f"- [[{sp.title}]]: {sp.description}"
            for sp in subpages
        )

        # ── 3. Chapter index via mini tool-augmented loop ─────────────
        # Run synchronously in the executor so we can share the event loop
        # with the sub-page tasks without nesting asyncio.run() calls.
        chapter_index_md = await loop.run_in_executor(
            None,
            lambda: self._generate_chapter_index(
                chapter_spec,
                evidence_slices,
                toc_wikilinks,
                subpage_mds=subpage_mds,
                stream_callback=stream_callback,
            ),
        )

        # ── 4. Inject cluster diagrams into the chapter index ─────────
        # Collect file paths from evidence so _inject can bypass the
        # skeleton cluster_id → DB macro_cluster mismatch.
        _evidence_paths: list[str] = []
        for _sl in evidence_slices:
            _evidence_paths.extend(getattr(_sl, "file_paths", []))

        # Run in executor: _inject_cluster_diagrams does sync disk IO and LLM
        # calls that would block the event loop if called directly.
        chapter_index_md = await loop.run_in_executor(
            None,
            lambda: self._inject_cluster_diagrams(
                chapter_index_md,
                chapter_spec.cluster_id,
                chapter_spec.chapter_title,
                file_hints=_evidence_paths,
            ),
        )

        logger.info(
            "[WRITER] generate_chapter done: title=%r subpages=%d index_chars=%d",
            chapter_spec.chapter_title,
            len(subpage_mds),
            len(chapter_index_md),
        )
        return chapter_index_md, subpage_mds

    def _inject_cluster_diagrams(
        self,
        chapter_index_md: str,
        cluster_id: int,
        chapter_title: str,
        file_hints: "list[str] | None" = None,
    ) -> str:
        """Inject Mermaid diagrams into a chapter _index.md using agentic file reading.

        Flow:
        1. Deterministic architecture diagram from graph (no LLM needed).
        2. LLM-based class diagram from actual source files.
        3. LLM-based data model diagram from actual source files.

        *file_hints* are relative paths from evidence slices — used to resolve
        the correct DB ``macro_cluster`` when skeleton cluster_ids diverge from
        stored ``macro_cluster`` values (they use different numbering systems).

        Sections are injected before ``## Key Components`` (or ``## Sub-pages``
        when Key Components is absent).  Returns original markdown unchanged
        when no diagrams are generated or on any error.
        """
        storage = self.tools.storage
        if storage is None:
            return chapter_index_md

        try:
            effective_llm = self.llm_client
            sections: list[str] = []

            db_cluster_id = self._resolve_db_cluster(cluster_id, storage, file_hints)
            logger.debug("[DIAGRAMS] cluster_id=%s → db_cluster_id=%s title=%r", cluster_id, db_cluster_id, chapter_title)

            # All three diagrams are generated by an agentic loop: the LLM
            # sees the available files and reads whichever it needs before
            # outputting each diagram type.
            if effective_llm is not None:
                available_files = self._list_cluster_files(
                    db_cluster_id, storage, file_hints=file_hints
                )
                if available_files:
                    arch = self._agentic_diagram(
                        available_files, chapter_title, "architecture"
                    )
                    if arch:
                        sections.append(f"## Architecture\n\n{arch}")

                    class_diag = self._agentic_diagram(
                        available_files, chapter_title, "class"
                    )
                    if class_diag:
                        sections.append(f"## Class Structure\n\n{class_diag}")

                    data_diag = self._agentic_diagram(
                        available_files, chapter_title, "datamodel"
                    )
                    if data_diag:
                        sections.append(f"## Data Model\n\n{data_diag}")

            if not sections:
                return chapter_index_md

            injected = "\n\n".join(sections)
            for marker in ("## Key Components", "## Sub-pages"):
                pos = chapter_index_md.find(marker)
                if pos >= 0:
                    return (
                        chapter_index_md[:pos].rstrip()
                        + f"\n\n{injected}\n\n"
                        + chapter_index_md[pos:]
                    )
            return chapter_index_md.rstrip() + f"\n\n{injected}"

        except Exception as exc:
            logger.debug("[WRITER] _inject_cluster_diagrams failed: %s", exc)
            return chapter_index_md

    def _resolve_db_cluster(
        self,
        cluster_id: int,
        storage: Any,
        file_hints: "list[str] | None",
    ) -> int:
        """Return the DB macro_cluster value that corresponds to this chapter.

        Skeleton cluster_ids (assigned 1-based by enumerate in structure_skeleton.py)
        differ from the 0-based macro_cluster values stored in the DB by the
        Leiden clustering step.  Probe using file_hints paths — the first node
        found for any hint path gives the true macro_cluster.  Falls back to
        cluster_id unchanged when probing fails or hints are absent.
        """
        if not file_hints or storage is None:
            return cluster_id
        try:
            from collections import Counter  # noqa: PLC0415
            hint_set = set(file_hints[:8])
            votes: Counter = Counter()
            # Query by directory but only count nodes whose rel_path is one of
            # the exact hint paths — avoids sibling subdirectory contamination.
            seen_dirs: set[str] = set()
            for path in file_hints[:8]:
                dir_prefix = path.rsplit("/", 1)[0] if "/" in path else path
                if dir_prefix in seen_dirs:
                    continue
                seen_dirs.add(dir_prefix)
                try:
                    nodes = storage.get_nodes_by_path_prefix(dir_prefix, limit=200)
                except Exception:
                    continue
                for n in nodes:
                    if n.get("rel_path") in hint_set:
                        mc = n.get("macro_cluster")
                        if mc is not None:
                            votes[int(mc)] += 1
            if votes:
                return votes.most_common(1)[0][0]
        except Exception as exc:
            logger.debug("[DIAGRAMS] _resolve_db_cluster failed: %s", exc)
        return cluster_id

    def _gather_cluster_source(
        self,
        cluster_id: int,
        storage: Any,
        max_files: int = 5,
        chars_per_file: int = 3000,
        file_hints: "list[str] | None" = None,
    ) -> str:
        """Read source files for cluster_id; return concatenated content for LLM context.

        Prefers *file_hints* (paths from evidence slices) over a DB cluster
        lookup, since skeleton cluster_ids don't always match DB macro_cluster
        values.  Falls back to cluster node query when hints are absent or
        produce no readable files.
        """
        _DOC_TYPES = {
            "config_document", "markdown_document", "toml_document",
            "module_doc", "file_doc", "rst_document", "html_document",
            "asciidoc_document",
        }

        def _read_paths(paths: list[str]) -> list[str]:
            parts: list[str] = []
            for path in paths[:max_files]:
                try:
                    fc = self.tools.read_file(path)
                    if fc.error or not fc.lines:
                        continue
                    content = "\n".join(fc.lines)[:chars_per_file]
                    parts.append(f"=== {path} ===\n{content}")
                except Exception:
                    continue
            return parts

        # Primary: use evidence file paths directly (avoids cluster_id mismatch).
        if file_hints:
            unique_hints = list(dict.fromkeys(p for p in file_hints if p))
            parts = _read_paths(unique_hints)
            if parts:
                return "\n\n".join(parts)

        # Fallback: query DB by cluster_id.
        try:
            nodes = storage.get_nodes_by_cluster(cluster_id) if storage else []
        except Exception:
            nodes = []

        code_nodes = [n for n in nodes if n.get("symbol_type") not in _DOC_TYPES]
        path_scores: dict[str, int] = {}
        for n in code_nodes:
            p = n.get("rel_path", "")
            if p:
                path_scores[p] = path_scores.get(p, 0) + (1 if n.get("is_architectural") else 0)

        sorted_paths = sorted(path_scores, key=lambda p: path_scores[p], reverse=True)
        return "\n\n".join(_read_paths(sorted_paths))

    def _list_cluster_files(
        self,
        db_cluster_id: int,
        storage: Any,
        file_hints: "list[str] | None" = None,
    ) -> list[str]:
        """Return deduplicated source file paths for this cluster.

        Prefers file_hints (from evidence slices) because they represent
        exactly what the planner considered relevant.  Falls back to querying
        the DB by macro_cluster and scoring by architectural node count.
        """
        _DOC_TYPES = {
            "config_document", "markdown_document", "toml_document",
            "module_doc", "file_doc", "rst_document", "html_document",
            "asciidoc_document",
        }

        # Primary: evidence paths are already the right files.
        if file_hints:
            unique = list(dict.fromkeys(p for p in file_hints if p))
            if unique:
                return unique

        # Fallback: query DB.
        try:
            nodes = storage.get_nodes_by_cluster(db_cluster_id) if storage else []
        except Exception:
            return []
        code_nodes = [n for n in nodes if n.get("symbol_type") not in _DOC_TYPES]
        scores: dict[str, int] = {}
        for n in code_nodes:
            p = n.get("rel_path", "")
            if p:
                scores[p] = scores.get(p, 0) + (1 if n.get("is_architectural") else 0)
        return sorted(scores, key=lambda p: scores[p], reverse=True)

    _DIAGRAM_BUDGET = 6  # max tool calls the LLM may make to read files

    _DIAGRAM_PROMPT_FILES: dict[str, str] = {
        "architecture": "diagrams/architecture.md",
        "class":        "diagrams/class_structure.md",
        "datamodel":    "diagrams/data_model.md",
    }

    _DIAGRAM_EXPECTED_HEADER: dict[str, tuple[str, ...]] = {
        "architecture": ("stateDiagram",),  # stateDiagram-v2 for functional flows
        "class": ("classDiagram",),
        "datamodel": ("erDiagram",),
    }

    def _agentic_diagram(
        self,
        available_files: list[str],
        chapter_title: str,
        diagram_type: str,
    ) -> str:
        """Run a tool-augmented loop to generate one Mermaid diagram.

        The LLM receives the list of available source files and calls
        ``read_file`` on whichever it needs (budget: _DIAGRAM_BUDGET calls).
        After reading it outputs the diagram.  Returns '' on failure/EMPTY.
        """
        if self.llm_client is None:
            return ""

        from langchain_core.messages import (  # noqa: PLC0415
            HumanMessage as _HM, SystemMessage as _SM, ToolMessage as _TM,
        )
        import re as _re  # noqa: PLC0415
        from .diagram_generator import _validate_mermaid  # noqa: PLC0415

        from app.prompts import load_prompt  # noqa: PLC0415
        prompt_file = self._DIAGRAM_PROMPT_FILES.get(diagram_type, "")
        system_prompt = load_prompt(prompt_file) if prompt_file else ""
        expected_headers = self._DIAGRAM_EXPECTED_HEADER.get(diagram_type, ())

        file_list = "\n".join(f"- {p}" for p in available_files)
        user_msg = (
            f"Chapter: {chapter_title}\n\n"
            f"Available source files:\n{file_list}\n\n"
            "Read the files you need and then output the diagram."
        )

        try:
            llm_with_tools = safe_bind_tools(self.llm_client, _WRITER_TOOL_SCHEMAS)
            messages: list[Any] = [
                _SM(content=system_prompt),
                _HM(content=user_msg),
            ]

            for _round in range(self._DIAGRAM_BUDGET + 1):
                response = llm_with_tools.invoke(messages)
                tool_calls = getattr(response, "tool_calls", None) or []

                if not tool_calls:
                    # No more tool calls — extract the diagram from the response.
                    text = _extract_llm_text(response)
                    if not text or text.strip() == "EMPTY":
                        return ""
                    m = _re.search(r"```mermaid\n.*?```", text, _re.DOTALL)
                    if not m:
                        return ""
                    candidate = m.group(0)
                    ok, _ = _validate_mermaid(candidate)
                    if not ok:
                        return ""
                    second_line = candidate.splitlines()[1] if len(candidate.splitlines()) > 1 else ""
                    if not any(h in second_line for h in expected_headers):
                        return ""
                    return candidate

                if _round >= self._DIAGRAM_BUDGET:
                    # Budget exhausted — force output now.
                    messages.append(response)
                    messages.append(_HM(content=(
                        "Budget reached. Output the diagram now based on what you have read. "
                        "Do not call any more tools."
                    )))
                    continue

                # Execute tool calls.
                messages.append(response)
                for tc in tool_calls:
                    tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    tc_id = tc.get("id", str(_round)) if isinstance(tc, dict) else getattr(tc, "id", str(_round))
                    if isinstance(tc_args, str):
                        try:
                            import json as _json; tc_args = _json.loads(tc_args)  # noqa: PLC0415,E702
                        except Exception:
                            tc_args = {}
                    result = self._dispatch_tool(tc_name, tc_args)
                    messages.append(_TM(content=result, tool_call_id=tc_id, name=tc_name))

        except Exception as exc:
            logger.debug("[WRITER] _agentic_diagram(%s) failed: %s", diagram_type, exc)

        return ""

    def _diagram_llm_invoke(self, prompt: str) -> str:
        """Invoke the LLM for diagram generation with tool calls suppressed.

        The writer's llm_client has WIKI_TOOL_SCHEMAS pre-bound so the model
        tends to call read_file instead of generating.  This helper:
        1. Tries to bind tool_choice='none' (Anthropic/OpenAI) to stop tool calls.
        2. Prepends a SystemMessage to reinforce the no-tools constraint.
        3. Falls back to a plain invoke if binding fails.
        """
        from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM  # noqa: PLC0415

        system_msg = _SM(content=(
            "You are a diagram generator. Output ONLY the requested Mermaid code block. "
            "Do not call any tools or functions. Do not request to read any files. "
            "Use only the source code provided in the user message."
        ))
        user_msg = _HM(content=prompt)

        # Try to suppress tool calls at the API level.
        llm = self.llm_client
        try:
            llm = llm.bind(tool_choice="none")
        except Exception:
            pass

        response = llm.invoke([system_msg, user_msg])
        return _extract_llm_text(response)

    def _llm_architecture_diagram(self, source_ctx: str, chapter_title: str) -> str:
        """Generate a Mermaid flowchart showing the architectural structure of this cluster.

        The LLM reads the source code and identifies meaningful components
        (services, handlers, repositories, clients, queues — not individual
        data fields or response model classes) and their relationships.
        Returns a validated fenced graph TD block, or '' on failure / empty output.
        """
        if self.llm_client is None:
            return ""

        prompt = (
            "The source code below is complete. Read it directly — do NOT call any tools "
            "or request additional files. Generate a Mermaid architecture diagram.\n\n"
            "Rules:\n"
            "- Show 4-8 meaningful architectural components: services, handlers, "
            "repositories, clients, queues, workers, routers\n"
            "- DO NOT include: individual response/request models, data classes, "
            "enums, constants, or utility functions\n"
            "- Label each edge with the relationship: calls, reads, writes, publishes, "
            "subscribes, manages, depends on\n"
            "- Use graph TD (top-down) layout\n"
            "- If the code has no meaningful architectural components (e.g. it is purely "
            "data models or utilities), return the string \"EMPTY\"\n"
            "- Return ONLY the ```mermaid\\ngraph TD\\n...``` block or the string \"EMPTY\"\n\n"
            f"Chapter: {chapter_title}\n\n"
            f"Source code:\n{source_ctx}"
        )

        try:
            text = self._diagram_llm_invoke(prompt)
            if not text or text.strip() == "EMPTY":
                return ""
            import re as _re  # noqa: PLC0415
            m = _re.search(r"```mermaid\n.*?```", text, _re.DOTALL)
            if not m:
                return ""
            from .diagram_generator import _validate_mermaid  # noqa: PLC0415
            candidate = m.group(0)
            ok, _ = _validate_mermaid(candidate)
            # Accept graph or flowchart headers only — reject classDiagram/erDiagram.
            second_line = candidate.splitlines()[1] if len(candidate.splitlines()) > 1 else ""
            if ok and not any(kw in second_line for kw in ("graph", "flowchart")):
                return ""
            return candidate if ok else ""
        except Exception as exc:
            logger.debug("[WRITER] _llm_architecture_diagram failed: %s", exc)
            return ""

    def _llm_class_diagram(self, source_ctx: str, chapter_title: str) -> str:
        """Use LLM to generate a Mermaid classDiagram from source code context.

        Returns a validated fenced classDiagram block, or '' on failure / empty output.
        """
        if self.llm_client is None:
            return ""

        prompt = (
            "The source code below is complete. Read it directly — do NOT call any tools "
            "or request additional files. Generate a Mermaid classDiagram immediately.\n\n"
            "Rules:\n"
            "- Include all significant classes, interfaces, abstract classes, enums, and structs\n"
            "- Show class members: +publicMethod(param: type): returnType, -_privateField, #protected\n"
            "- Show inheritance: ChildClass --|> ParentClass\n"
            "- Show composition: ClassA *-- ClassB (when A contains B as a field)\n"
            "- Add stereotypes: <<interface>>, <<abstract>>, <<enumeration>>\n"
            "- Skip test classes, private helpers, and trivial single-method classes\n"
            "- Cap at 10 classes maximum; prefer the most important ones\n"
            "- If there are fewer than 2 meaningful classes, return the string \"EMPTY\"\n"
            "- Return ONLY the ```mermaid classDiagram ... ``` block or the string \"EMPTY\"\n\n"
            f"Chapter: {chapter_title}\n\n"
            f"Source code:\n{source_ctx}"
        )

        try:
            text = self._diagram_llm_invoke(prompt)
            if not text or text.strip() == "EMPTY":
                return ""
            import re as _re  # noqa: PLC0415
            m = _re.search(r"```mermaid\n.*?```", text, _re.DOTALL)
            if not m:
                return ""
            from .diagram_generator import _validate_mermaid  # noqa: PLC0415
            candidate = m.group(0)
            ok, _ = _validate_mermaid(candidate)
            if ok and "classDiagram" not in candidate.splitlines()[1]:
                return ""
            return candidate if ok else ""
        except Exception as exc:
            logger.debug("[WRITER] _llm_class_diagram failed: %s", exc)
            return ""

    def _llm_data_model_diagram(self, source_ctx: str, chapter_title: str) -> str:
        """Use LLM to generate a Mermaid erDiagram from source code context.

        Only produces a diagram for real database/persistence models (SQLAlchemy,
        Django, TypeORM, GORM, etc.) — not Pydantic schemas or plain dataclasses.
        Returns a validated fenced erDiagram block, or '' on failure / empty output.
        """
        if self.llm_client is None:
            return ""

        prompt = (
            "The source code below is complete. Read it directly — do NOT call any tools "
            "or request additional files. Identify ONLY true database/persistence "
            "models — not API response schemas.\n\n"
            "Database models include:\n"
            "- SQLAlchemy ORM models (classes with Column(), relationship(), mapped_column())\n"
            "- Django models (models.Model subclasses)\n"
            "- TypeORM entities (@Entity decorator)\n"
            "- Go GORM structs (gorm: tags)\n"
            "- Prisma schema definitions\n\n"
            "DO NOT include: Pydantic response/request models, dataclasses used as DTOs "
            "or value objects, plain TypeScript interfaces, configuration classes.\n\n"
            "If you find 2 or more database models, produce a Mermaid erDiagram showing:\n"
            "- Entity names and their fields with types (int, str, uuid, etc.)\n"
            "- Mark primary keys with PK, foreign keys with FK\n"
            "- Relationships between entities (||--o{, }|--|{, etc.)\n\n"
            "If fewer than 2 database models exist, return the string \"EMPTY\".\n"
            "Return ONLY the ```mermaid erDiagram ... ``` block or the string \"EMPTY\".\n\n"
            f"Chapter: {chapter_title}\n\n"
            f"Source code:\n{source_ctx}"
        )

        try:
            text = self._diagram_llm_invoke(prompt)
            if not text or text.strip() == "EMPTY":
                return ""
            import re as _re  # noqa: PLC0415
            m = _re.search(r"```mermaid\n.*?```", text, _re.DOTALL)
            if not m:
                return ""
            from .diagram_generator import _validate_mermaid  # noqa: PLC0415
            candidate = m.group(0)
            ok, _ = _validate_mermaid(candidate)
            # Enforce erDiagram header — reject graph LR or classDiagram blocks.
            if ok and "erDiagram" not in candidate.splitlines()[1]:
                return ""
            return candidate if ok else ""
        except Exception as exc:
            logger.debug("[WRITER] _llm_data_model_diagram failed: %s", exc)
            return ""

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
