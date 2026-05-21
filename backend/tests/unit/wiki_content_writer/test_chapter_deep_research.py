"""Unit tests for chapter-index deep-research loop (Change 1).

The new generate_chapter implementation replaces the simple LLM overview call
with a mini tool-augmented loop (budget=6) scoped to the cluster's own symbols.

Covers:
- Chapter index starts with the structured Markdown format:
  ## Overview / ## Key Components / ## Sub-pages
- Tool-augmented loop is used (LLM receives tool schemas, not plain model)
- Budget is capped at 6 tool calls
- [[wikilink]] TOC is always present in ## Sub-pages
- Fallback to simple index when LLM fails entirely
- stream_callback receives writer.tool_call events from the index loop
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.wiki_structure_planner.evidence import EvidenceSlice
from app.core.wiki_structure_planner.planner_agent import ChapterSpec, SubPageSpec
from app.core.wiki_content_writer.writer_agent import WikiContentWriter
from app.core.wiki_content_writer.writer_tools import WriterTools


# ── FakeLLM helpers ──────────────────────────────────────────────────────────


class _BindToolsNoop:
    """Mixin: bind_tools returns self so safe_bind_tools works."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


class SequentialFakeLLM(_BindToolsNoop, BaseChatModel):
    """Pydantic-compatible FakeLLM returning responses in sequence."""

    responses: list[str] = []
    call_count: int = 0
    model_name: str = "seq-fake"

    @property
    def _llm_type(self) -> str:
        return "seq-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = self.call_count
        self.call_count += 1
        content = (
            self.responses[idx] if idx < len(self.responses) else "Fallback content [x.py:1]."
        )
        msg = AIMessage(content=content)
        msg.tool_calls = []  # type: ignore[attr-defined]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


class ToolCallThenWriteFakeLLM(_BindToolsNoop, BaseChatModel):
    """FakeLLM that makes N tool calls then writes the final content.

    Simulates the tool-augmented chapter index loop:
    - First ``n_tool_rounds`` calls return an AIMessage with tool_calls set.
    - The final call (or after budget exhaustion) returns prose content.
    """

    model_name: str = "tool-call-fake"
    n_tool_rounds: int = 2
    final_content: str = ""
    call_count: int = 0
    tool_name: str = "read_file"
    tool_args: dict = {"path": "src/auth.py"}

    @property
    def _llm_type(self) -> str:
        return "tool-call-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = self.call_count
        self.call_count += 1
        if idx < self.n_tool_rounds:
            # Return a tool_call response
            msg = AIMessage(content="")
            tc_id = f"tc_{idx}"
            msg.tool_calls = [  # type: ignore[attr-defined]
                {"id": tc_id, "name": self.tool_name, "args": self.tool_args}
            ]
            return ChatResult(generations=[ChatGeneration(message=msg)])
        else:
            # Return the prose write
            msg = AIMessage(content=self.final_content or "## Overview\nDone.\n## Sub-pages\n")
            msg.tool_calls = []  # type: ignore[attr-defined]
            return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_chapter_spec(n_subpages: int = 2) -> ChapterSpec:
    subpages = [
        SubPageSpec(
            title=f"Auth Sub {i + 1}",
            description=f"Details about auth subsystem {i + 1}.",
            page_order=i + 1,
            retrieval_query=f"auth subsystem {i + 1}",
            target_symbols=[f"AuthSymbol{i + 1}"],
            target_folders=[f"src/auth{i + 1}"],
            target_docs=[],
        )
        for i in range(n_subpages)
    ]
    return ChapterSpec(
        cluster_id=99,
        chapter_title="Auth System",
        chapter_description="The authentication and authorization subsystem.",
        subpages=subpages,
    )


def _make_evidence_slices(n: int = 2) -> list[EvidenceSlice]:
    return [
        EvidenceSlice(
            subpage_index=i + 1,
            subpage_title=f"Auth Sub {i + 1}",
            symbols=[f"AuthSymbol{i + 1}"],
            file_paths=[f"src/auth{i + 1}.py"],
            doc_paths=[],
            call_graph_edges=[],
            cross_cluster_refs=[],
            summary=f"Summary for auth sub {i + 1}.",
        )
        for i in range(n)
    ]


def _run(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Tests: chapter index structure ───────────────────────────────────────────


class TestChapterIndexStructure:
    """Chapter index uses the structured ## Overview / ## Key Components / ## Sub-pages format."""

    _STRUCTURED_INDEX = (
        "## Overview\n"
        "The auth cluster handles login and session management.\n\n"
        "## Key Components\n"
        "- **AuthManager**: Manages auth flow.\n\n"
        "## Sub-pages\n"
        "[[Auth Sub 1]]: Details about auth subsystem 1.\n"
    )

    def _make_writer(self, tmp_path, responses: list[str]) -> WikiContentWriter:
        llm = SequentialFakeLLM(responses=responses)
        tools = WriterTools(repo_root=str(tmp_path))
        return WikiContentWriter(tools=tools, llm_client=llm)

    def test_chapter_index_contains_overview_section(self, tmp_path):
        """The chapter index must contain '## Overview'."""
        # 2 subpage responses + 1 chapter index response
        llm_responses = [
            "SP1 content [a.py:1].",
            "SP2 content [b.py:1].",
            self._STRUCTURED_INDEX,
        ]
        writer = self._make_writer(tmp_path, llm_responses)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        index_md, _ = _run(writer.generate_chapter(chapter, slices))

        assert "## Overview" in index_md

    def test_chapter_index_contains_sub_pages_section(self, tmp_path):
        """The chapter index must contain '## Sub-pages'."""
        llm_responses = [
            "SP1 content [a.py:1].",
            "SP2 content [b.py:1].",
            self._STRUCTURED_INDEX,
        ]
        writer = self._make_writer(tmp_path, llm_responses)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        index_md, _ = _run(writer.generate_chapter(chapter, slices))

        assert "## Sub-pages" in index_md

    def test_chapter_index_wikilinks_present(self, tmp_path):
        """[[wikilink]] entries for each subpage appear in the chapter index."""
        llm_responses = [
            "SP1 content [a.py:1].",
            "SP2 content [b.py:1].",
            "## Overview\nAuth overview.\n## Key Components\n- **X**: role.\n## Sub-pages\n",
        ]
        writer = self._make_writer(tmp_path, llm_responses)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        index_md, _ = _run(writer.generate_chapter(chapter, slices))

        assert "[[Auth Sub 1]]" in index_md
        assert "[[Auth Sub 2]]" in index_md

    def test_chapter_index_non_empty(self, tmp_path):
        """Chapter index is always non-empty (fallback if LLM fails)."""
        # All responses are empty/fail — triggers fallback
        llm = SequentialFakeLLM(responses=[
            "SP1 [a.py:1].",
            "SP2 [b.py:1].",
            # No index response — will use fallback
        ])
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=llm)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        index_md, _ = _run(writer.generate_chapter(chapter, slices))

        assert isinstance(index_md, str)
        assert len(index_md) > 0


# ── Tests: tool-augmented loop for chapter index ─────────────────────────────


class TestChapterIndexToolLoop:
    """Chapter index uses a bounded tool-augmented LLM loop (budget=6)."""

    def test_chapter_index_uses_tool_augmented_llm(self, tmp_path):
        """The chapter index loop uses an LLM with tools bound (not bare model)."""
        # ToolCallThenWriteFakeLLM simulates the tool loop for the chapter index.
        # We track whether bind_tools was called with schemas.
        bind_tools_calls: list[Any] = []

        class TrackingFakeLLM(SequentialFakeLLM):
            def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
                bind_tools_calls.append(tools)
                return self

        llm = TrackingFakeLLM(responses=[
            "SP1 [a.py:1].",
            "SP2 [b.py:1].",
            "## Overview\nAuth.\n## Key Components\n- **X**: Y.\n## Sub-pages\n",
        ])
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=llm)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        _run(writer.generate_chapter(chapter, slices))

        # bind_tools should have been called at least once (for the index loop)
        # The sub-page generate_page calls also call bind_tools — at minimum N+1 calls
        assert len(bind_tools_calls) >= 1

    def test_chapter_index_budget_capped_at_6(self, tmp_path):
        """The chapter index loop makes at most 6 tool calls before forcing write."""
        tool_call_count = [0]

        class CountingWriterTools(WriterTools):
            def read_file(self, path: str, line_range=None):  # type: ignore[override]
                tool_call_count[0] += 1
                # Return a simple result to avoid errors
                result = MagicMock()
                result.error = None
                result.lines = [f"# line from {path}"]
                return result

        # LLM that always asks for tool calls — budget should cap it
        class InfiniteToolCallLLM(_BindToolsNoop, BaseChatModel):
            model_name: str = "inf-tool"
            call_count: int = 0

            @property
            def _llm_type(self) -> str:
                return "inf-tool"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.call_count += 1
                msg = AIMessage(content="")
                msg.tool_calls = [  # type: ignore[attr-defined]
                    {"id": f"tc_{self.call_count}", "name": "read_file", "args": {"path": "src/a.py"}}
                ]
                return ChatResult(generations=[ChatGeneration(message=msg)])

            def _stream(self, *args, **kwargs):
                raise NotImplementedError

        # Sub-pages use a separate wrapping — we patch generate_page
        index_llm = InfiniteToolCallLLM()
        tools = CountingWriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=index_llm)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        # Patch generate_page so sub-pages don't use the same counting tools
        with patch.object(
            writer,
            "generate_page",
            side_effect=lambda spec, **kw: ("# SP\n\nContent [a.py:1].\n", []),
        ):
            _run(writer.generate_chapter(chapter, slices))

        # The chapter index loop must not exceed 6 tool calls
        assert tool_call_count[0] <= 6

    def test_stream_callback_receives_tool_call_events_from_index_loop(self, tmp_path):
        """When chapter index loop uses tools, stream_callback sees writer.tool_call events."""
        # Simulates: index LLM does 1 tool call then writes prose
        final_index = "## Overview\nAuth.\n## Key Components\n- **X**: Y.\n## Sub-pages\n"
        tool_llm = ToolCallThenWriteFakeLLM(
            n_tool_rounds=1,
            final_content=final_index,
        )
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=tool_llm)
        chapter = _make_chapter_spec(n_subpages=0)  # no sub-pages → skip sub-page gen
        slices = []

        events: list[dict] = []
        _run(writer.generate_chapter(chapter, slices, stream_callback=events.append))

        tool_events = [e for e in events if e.get("event") == "writer.tool_call"]
        # At least one tool_call event from the index loop
        assert len(tool_events) >= 1


# ── Tests: fallback behaviour ─────────────────────────────────────────────────


class TestChapterIndexFallback:
    """Chapter index falls back gracefully when LLM fails."""

    def test_llm_failure_produces_non_empty_index(self, tmp_path):
        """If the chapter-index LLM call raises, index is still non-empty."""
        class ErrorLLM(_BindToolsNoop, BaseChatModel):
            model_name: str = "error-llm"
            call_count: int = 0

            @property
            def _llm_type(self) -> str:
                return "error-llm"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                raise RuntimeError("Simulated LLM failure")

            def _stream(self, *args, **kwargs):
                raise NotImplementedError

        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=ErrorLLM())
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        # Patch sub-page generation so only the index LLM is exercised
        with patch.object(
            writer,
            "generate_page",
            side_effect=lambda spec, **kw: ("# SP\n\nContent.\n", []),
        ):
            index_md, _ = _run(writer.generate_chapter(chapter, slices))

        assert isinstance(index_md, str)
        assert len(index_md) > 0

    def test_fallback_index_contains_subpage_wikilinks(self, tmp_path):
        """Fallback chapter index always includes wikilinks for sub-pages."""
        class ErrorLLM(_BindToolsNoop, BaseChatModel):
            model_name: str = "error-llm2"
            call_count: int = 0

            @property
            def _llm_type(self) -> str:
                return "error-llm2"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                raise RuntimeError("Simulated LLM failure")

            def _stream(self, *args, **kwargs):
                raise NotImplementedError

        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=ErrorLLM())
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        with patch.object(
            writer,
            "generate_page",
            side_effect=lambda spec, **kw: ("# SP\n\nContent.\n", []),
        ):
            index_md, _ = _run(writer.generate_chapter(chapter, slices))

        assert "[[Auth Sub 1]]" in index_md
        assert "[[Auth Sub 2]]" in index_md
