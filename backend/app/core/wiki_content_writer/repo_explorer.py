"""Agentic repository deep exploration (#feat — repo analysis upgrade).

Runs a tool-augmented LLM loop that reads the actual codebase before
producing a structured architectural analysis document. Replaces the
single-pass ``_llm_analyze_repository`` call.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from app.core.agent_tools import WIKI_TOOL_SCHEMAS
from app.services.llm_factory import is_bedrock_llm, make_cached_system_message, safe_bind_tools

from .writer_tools import WriterTools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior software architect doing a first-day deep dive on a new codebase.
Your goal: produce a comprehensive architectural analysis document.

You have tools to read the actual files. Use them systematically:
1. Read entry points (main.py, app.py, __init__.py, run.py)
2. Follow imports to understand the application structure
3. Read router/controller files to understand the API surface
4. Read main service/domain files to understand business logic
5. Read model/schema files to understand data contracts
6. Read config files to understand external dependencies

REQUIRED OUTPUT STRUCTURE (fill all sections):
# Repository Analysis: {repo_name}

## Purpose
What does this software do? What problem does it solve?

## Target Users & Use Cases
Who uses this? How?

## Architecture
Key components and their responsibilities.

## Main Data Flows
Step-by-step description of 2-3 key flows.

## Technology Stack
Languages, frameworks, databases, external services.

## Key Design Decisions
Notable patterns, architectural choices, trade-offs.

## Integration Points
APIs exposed, external services consumed.

Use your tool budget ({budget} calls) wisely. Start broad (entry points),
go deep on the most important files.
"""

_BUDGET_EXHAUSTED_PROMPT = (
    "You have used all available tool calls. Based on the files you explored, "
    "write the complete architectural analysis now following the REQUIRED OUTPUT STRUCTURE. "
    "Include all seven sections. Do NOT call any more tools."
)


class RepoExplorer:
    """Agentic repository explorer that reads the codebase before analysing.

    Parameters
    ----------
    tools : WriterTools
        Tool surface for file and symbol access.
    llm_client : BaseChatModel | None
        Pre-configured LangChain chat model (primary LLM, not low-tier).
    storage : WikiStorageProtocol | None
        Optional storage for symbol graph queries (search_symbols,
        list_cluster_symbols).  Taken from ``tools.storage`` when not
        supplied explicitly.
    """

    def __init__(
        self,
        tools: WriterTools,
        llm_client: Any | None = None,
        storage: Any | None = None,
    ) -> None:
        self.tools = tools
        self.llm_client = llm_client
        # Prefer explicit storage; fall back to the one embedded in tools.
        self.storage = storage if storage is not None else getattr(tools, "storage", None)

    # ── Extra graph tools ─────────────────────────────────────────────────

    def _search_symbols(self, query: str) -> str:
        """Search the code graph for symbols matching query."""
        if self.storage is None:
            return "[no storage]"
        try:
            results = self.storage.search_fts_by_symbol_name(query, limit=10)
            if not results:
                return "[no matches]"
            return "\n".join(
                f"{r.get('symbol_name')} @ {r.get('rel_path')}:{r.get('start_line')}"
                for r in results
            )
        except Exception as exc:
            logger.debug("search_symbols failed for %r: %s", query, exc)
            return f"[error: {exc}]"

    def _list_cluster_symbols(self, cluster_id: int) -> str:
        """List architectural symbols in a cluster."""
        if self.storage is None:
            return "[no storage]"
        try:
            nodes = self.storage.get_nodes_by_cluster(cluster_id, limit=30)
            if not nodes:
                return "[no symbols in cluster]"
            return "\n".join(
                f"{n.get('symbol_name')} ({n.get('symbol_type')}) @ {n.get('rel_path')}"
                for n in nodes
            )
        except Exception as exc:
            logger.debug("list_cluster_symbols failed for cluster_id=%d: %s", cluster_id, exc)
            return f"[error: {exc}]"

    # ── Tool dispatch ─────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        """Dispatch a single tool call and return the result as a string."""
        try:
            if name == "read_file":
                path = args.get("path", "")
                start_line = args.get("start_line")
                end_line = args.get("end_line")
                line_range = None
                if start_line is not None and end_line is not None:
                    line_range = (int(start_line), int(end_line))
                fc = self.tools.read_file(path, line_range=line_range)
                if fc.error:
                    return f"[error] {fc.error}"
                numbered = "\n".join(
                    f"{i + (line_range[0] if line_range else 1)}: {ln}"
                    for i, ln in enumerate(fc.lines)
                )
                return numbered or "[empty file]"

            elif name == "get_signature":
                sig = self.tools.get_signature(args.get("symbol", ""))
                if not sig.found:
                    return f"[not found] {args.get('symbol', '')}"
                return f"{sig.file_path}: {sig.signature} ({sig.layer})\n{sig.docstring}"

            elif name == "grep":
                matches = self.tools.grep(args.get("pattern", ""))
                if not matches:
                    return "[no matches]"
                return "\n".join(
                    f"{m.file_path}:{m.line_number}: {m.line_text}" for m in matches
                )

            elif name == "get_callers":
                result = self.tools.get_callers(args.get("symbol", ""))
                if not result.found:
                    return f"[not found] {args.get('symbol', '')}"
                if not result.callers:
                    return "[no callers]"
                return "\n".join(
                    f"{c.file_path}: {c.symbol}" for c in result.callers
                )

            elif name == "get_callees":
                result = self.tools.get_callees(args.get("symbol", ""))
                if not result.found:
                    return f"[not found] {args.get('symbol', '')}"
                if not result.callees:
                    return "[no callees]"
                return "\n".join(
                    f"{c.file_path}: {c.symbol}" for c in result.callees
                )

            elif name == "search_symbols":
                return self._search_symbols(args.get("query", ""))

            elif name == "list_cluster_symbols":
                return self._list_cluster_symbols(int(args.get("cluster_id", 0)))

            elif name == "list_doc_chunks":
                chunks = self.tools.list_doc_chunks(args.get("doc_path", ""))
                if not chunks:
                    return "[no chunks found]"
                return "\n".join(
                    f"[{c.chunk_index}] {c.heading}\n{c.text}" for c in chunks
                )

            else:
                return f"[unknown tool: {name}]"

        except Exception as exc:
            logger.debug("RepoExplorer tool dispatch failed name=%r: %s", name, exc)
            return f"[tool error: {exc}]"

    # ── Main explore loop ─────────────────────────────────────────────────

    def explore(
        self,
        file_tree: str,
        readme: str,
        repo_name: str,
        budget: int = 25,
    ) -> str:
        """Return a structured architectural analysis document.

        Parameters
        ----------
        file_tree : str
            Repository file tree (newline-separated paths).
        readme : str
            Contents of the repository README (or empty string).
        repo_name : str
            Repository name used in the analysis heading.
        budget : int
            Maximum number of tool calls before forcing a write.

        Returns
        -------
        str
            Markdown architectural analysis document.  Falls back to a
            minimal placeholder if the LLM is unavailable or always fails.
        """
        if self.llm_client is None:
            return self._fallback(repo_name)

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            repo_name=repo_name,
            budget=budget,
        )

        # Build the extra explorer schemas (search_symbols + list_cluster_symbols)
        _extra_schemas: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "search_symbols",
                    "description": "Search the code graph for symbols matching a query string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Symbol name or keyword."},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_cluster_symbols",
                    "description": "List architectural symbols in a given cluster.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cluster_id": {"type": "integer", "description": "Cluster ID."},
                        },
                        "required": ["cluster_id"],
                    },
                },
            },
        ]

        all_schemas = list(WIKI_TOOL_SCHEMAS) + _extra_schemas

        try:
            llm_with_tools = safe_bind_tools(
                self.llm_client,
                all_schemas,
            )
        except Exception as exc:
            logger.warning("RepoExplorer: safe_bind_tools failed — falling back to raw LLM: %s", exc)
            llm_with_tools = self.llm_client

        user_prompt = (
            f"Repository: {repo_name}\n\n"
            f"## File Tree\n{file_tree[:3000]}\n\n"
            f"## README\n{readme[:2000] if readme else 'No README found.'}\n\n"
            "Start by reading the key entry points, then explore the architecture. "
            "Once you have enough context, write the complete analysis."
        )

        _use_cache = is_bedrock_llm(self.llm_client)
        messages: list[Any] = [
            make_cached_system_message(system_prompt, cache=_use_cache),
            HumanMessage(content=user_prompt),
        ]

        tool_calls_made = 0
        content = ""
        # Cap rounds at budget (loose upper bound — actual cap is tool_calls_made >= budget).
        _MAX_ROUNDS = budget + 5

        try:
            for _round in range(_MAX_ROUNDS):
                response = llm_with_tools.invoke(messages)
                tool_calls = getattr(response, "tool_calls", None) or []

                if not tool_calls:
                    # LLM wrote the analysis.
                    raw = response.content
                    content = raw if isinstance(raw, str) else str(raw)
                    break

                # Execute all tool calls in this response.
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
                    messages.append(
                        ToolMessage(content=result_text, tool_call_id=tc_id, name=tc_name)
                    )

                logger.debug(
                    "[REPO_EXPLORER] round %d/%d: %d tool calls this round, %d total",
                    _round + 1,
                    budget,
                    len(tool_calls),
                    tool_calls_made,
                )

                if tool_calls_made >= budget:
                    break

            if not content:
                # Budget exhausted — force a write.
                messages.append(HumanMessage(content=_BUDGET_EXHAUSTED_PROMPT))
                try:
                    response = self.llm_client.invoke(messages)
                    raw = response.content
                    content = raw if isinstance(raw, str) else str(raw)
                except Exception as exc:
                    logger.warning("[REPO_EXPLORER] forced write call failed: %s", exc)

        except Exception as exc:
            logger.error("[REPO_EXPLORER] explore loop failed: %s", exc, exc_info=True)

        if not content:
            return self._fallback(repo_name)

        logger.info(
            "[REPO_EXPLORER] analysis complete: %d chars, %d tool calls",
            len(content),
            tool_calls_made,
        )
        return content

    @staticmethod
    def _fallback(repo_name: str) -> str:
        """Return a minimal analysis document when the LLM is unavailable."""
        return (
            f"# Repository Analysis: {repo_name}\n\n"
            "## Purpose\nRepository analysis unavailable.\n\n"
            "## Architecture\nSee the generated wiki pages for architectural details.\n"
        )
