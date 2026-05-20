"""Unit tests for WikiContentWriter.generate_chapter (#290).

Covers:
- generate_chapter returns (chapter_index_md, [sp1_md, sp2_md]) with 2 sub-pages
- sub-page oversized (>250 lines) emits writer.subpage_oversized stream event
- chapter index contains [[SubPage Title]] wikilinks for all sub-pages
- fallback markdown is used when one sub-page generation raises
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

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
    """Mixin: bind_tools returns self so writer_agent's safe_bind_tools works."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


class SequentialFakeLLM(_BindToolsNoop, BaseChatModel):
    """FakeLLM that returns responses from a list in order.

    Each call pops the next response. After the list is exhausted, returns
    a generic fallback.
    """

    responses: list[str] = []
    call_count: int = 0
    model_name: str = "sequential-fake"

    @property
    def _llm_type(self) -> str:
        return "sequential-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = self.call_count
        self.call_count += 1
        if idx < len(self.responses):
            content = self.responses[idx]
        else:
            content = f"Fallback page content [dummy.py:1]."
        msg = AIMessage(content=content)
        msg.tool_calls = []  # type: ignore[attr-defined]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


class DirectWriterFakeLLM(_BindToolsNoop, BaseChatModel):
    """FakeLLM that always returns the same content string."""

    response: str = ""
    model_name: str = "direct-writer-fake"

    @property
    def _llm_type(self) -> str:
        return "direct-writer-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(content=self.response)
        msg.tool_calls = []  # type: ignore[attr-defined]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_chapter_spec(n_subpages: int = 2) -> ChapterSpec:
    """Build a minimal ChapterSpec with n sub-pages."""
    subpages = [
        SubPageSpec(
            title=f"SubPage {i + 1}",
            description=f"Description for sub-page {i + 1}.",
            page_order=i + 1,
            retrieval_query=f"query sub {i + 1}",
            target_symbols=[f"Symbol{i + 1}"],
            target_folders=[f"src/module{i + 1}"],
            target_docs=[],
        )
        for i in range(n_subpages)
    ]
    return ChapterSpec(
        cluster_id=42,
        chapter_title="Auth Chapter",
        chapter_description="Authentication and authorization subsystem.",
        subpages=subpages,
    )


def _make_evidence_slices(n: int = 2) -> list[EvidenceSlice]:
    """Build minimal EvidenceSlice list with n entries."""
    return [
        EvidenceSlice(
            subpage_index=i + 1,
            subpage_title=f"SubPage {i + 1}",
            symbols=[f"Symbol{i + 1}"],
            file_paths=[f"src/module{i + 1}.py"],
            doc_paths=[],
            call_graph_edges=[],
            cross_cluster_refs=[],
            summary=f"Summary for sub-page {i + 1}.",
        )
        for i in range(n)
    ]


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously in tests.

    Always creates a new event loop so tests are independent of pytest-asyncio's
    loop lifecycle.  This avoids "Event loop is closed" errors when async tests
    from other modules run before these sync-helper tests.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Tests: generate_chapter returns correct structure ────────────────────────


class TestGenerateChapterBasic:
    """generate_chapter returns (index_md, [sp1_md, sp2_md]) for 2 sub-pages."""

    _SP1_MD = "Sub-page 1 content with real code. [src/module1.py:10]"
    _SP2_MD = "Sub-page 2 content with real code. [src/module2.py:20]"
    # index is the last call
    _INDEX_MD = "Chapter overview and TOC. [[SubPage 1]]: desc1\n[[SubPage 2]]: desc2"

    def _make_writer(self, tmp_path) -> WikiContentWriter:
        # 3 responses: sp1, sp2, then chapter index
        llm = SequentialFakeLLM(responses=[self._SP1_MD, self._SP2_MD, self._INDEX_MD])
        tools = WriterTools(repo_root=str(tmp_path))
        return WikiContentWriter(tools=tools, llm_client=llm)

    def test_returns_tuple_of_index_and_subpages(self, tmp_path):
        writer = self._make_writer(tmp_path)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        result = _run(writer.generate_chapter(chapter, slices))

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_subpages_list_has_correct_length(self, tmp_path):
        writer = self._make_writer(tmp_path)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        _, subpage_mds = _run(writer.generate_chapter(chapter, slices))

        assert len(subpage_mds) == 2

    def test_subpages_are_strings(self, tmp_path):
        writer = self._make_writer(tmp_path)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        _, subpage_mds = _run(writer.generate_chapter(chapter, slices))

        for md in subpage_mds:
            assert isinstance(md, str)

    def test_index_md_is_string(self, tmp_path):
        writer = self._make_writer(tmp_path)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        index_md, _ = _run(writer.generate_chapter(chapter, slices))

        assert isinstance(index_md, str)
        assert index_md  # non-empty

    def test_subpage_content_comes_from_llm(self, tmp_path):
        """Sub-pages get the markdown content returned by the LLM."""
        writer = self._make_writer(tmp_path)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        _, subpage_mds = _run(writer.generate_chapter(chapter, slices))

        # Both sub-pages should have non-empty content
        for md in subpage_mds:
            assert len(md) > 0

    def test_no_stream_callback_does_not_error(self, tmp_path):
        """generate_chapter works without a stream_callback."""
        writer = self._make_writer(tmp_path)
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        # Should not raise
        result = _run(writer.generate_chapter(chapter, slices, stream_callback=None))
        assert result is not None


# ── Tests: chapter index contains [[wikilinks]] ──────────────────────────────


class TestGenerateChapterIndex:
    """Chapter index must contain [[SubPage Title]] wikilinks."""

    def _make_writer(self, tmp_path, subpage_responses: list[str], index_response: str) -> WikiContentWriter:
        responses = subpage_responses + [index_response]
        llm = SequentialFakeLLM(responses=responses)
        tools = WriterTools(repo_root=str(tmp_path))
        return WikiContentWriter(tools=tools, llm_client=llm)

    def test_chapter_index_contains_wikilinks_for_all_subpages(self, tmp_path):
        """Chapter index markdown must include [[SubPage Title]] for each sub-page."""
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        # Index response from LLM will be appended with the TOC by generate_chapter
        index_response = "Brief chapter overview. [dummy.py:1]"
        writer = self._make_writer(
            tmp_path,
            subpage_responses=["SP1 content [src/a.py:1].", "SP2 content [src/b.py:1]."],
            index_response=index_response,
        )

        index_md, _ = _run(writer.generate_chapter(chapter, slices))

        # Chapter index must contain wikilinks for both sub-pages
        assert "[[SubPage 1]]" in index_md
        assert "[[SubPage 2]]" in index_md

    def test_chapter_index_wikilinks_use_exact_subpage_titles(self, tmp_path):
        """Wikilinks in index use the exact titles from SubPageSpec."""
        subpages = [
            SubPageSpec(
                title="Session Management",
                description="Handles session lifecycle.",
                page_order=1,
                retrieval_query="session",
                target_symbols=[],
                target_folders=[],
                target_docs=[],
            ),
            SubPageSpec(
                title="Token Validation",
                description="Validates JWT tokens.",
                page_order=2,
                retrieval_query="token",
                target_symbols=[],
                target_folders=[],
                target_docs=[],
            ),
        ]
        chapter = ChapterSpec(
            cluster_id=7,
            chapter_title="Auth System",
            chapter_description="Full auth system docs.",
            subpages=subpages,
        )
        slices = _make_evidence_slices(n=2)

        writer = self._make_writer(
            tmp_path,
            subpage_responses=["Session content [src/s.py:1].", "Token content [src/t.py:1]."],
            index_response="Auth system overview. [src/a.py:1]",
        )

        index_md, _ = _run(writer.generate_chapter(chapter, slices))

        assert "[[Session Management]]" in index_md
        assert "[[Token Validation]]" in index_md

    def test_chapter_index_contains_chapter_description(self, tmp_path):
        """The chapter title or description appears in the index prompt context."""
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        # The LLM response is used in the index; verify that the prompt was built
        # with the chapter description (by checking the result isn't just empty)
        llm_responses = [
            "SP1 content [a.py:1].",
            "SP2 content [b.py:1].",
            "Auth Chapter overview text. [c.py:1]",
        ]
        llm = SequentialFakeLLM(responses=llm_responses)
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=llm)

        index_md, _ = _run(writer.generate_chapter(chapter, slices))

        # Index is non-empty and contains something from the LLM response
        assert len(index_md) > 0


# ── Tests: oversized sub-page SSE event ──────────────────────────────────────


class TestSubPageOversizedEvent:
    """Sub-pages over 250 lines emit writer.subpage_oversized stream event."""

    _OVERSIZED_MD = "\n".join(f"Line {i}" for i in range(260))  # 260 lines > 250
    _NORMAL_MD = "Normal content [src/b.py:1]."

    def _make_writer(self, tmp_path, responses: list[str]) -> WikiContentWriter:
        llm = SequentialFakeLLM(responses=responses)
        tools = WriterTools(repo_root=str(tmp_path))
        return WikiContentWriter(tools=tools, llm_client=llm)

    def test_oversized_subpage_emits_event(self, tmp_path):
        """A sub-page with more than 250 lines triggers writer.subpage_oversized."""
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        writer = self._make_writer(
            tmp_path,
            responses=[
                self._OVERSIZED_MD,  # sp1 > 250 lines
                self._NORMAL_MD,     # sp2 normal
                "Chapter index [idx.py:1]",  # chapter index
            ],
        )

        events: list[dict] = []
        _run(writer.generate_chapter(chapter, slices, stream_callback=events.append))

        oversized_events = [e for e in events if e.get("event") == "writer.subpage_oversized"]
        assert len(oversized_events) >= 1

    def test_oversized_event_contains_title(self, tmp_path):
        """The writer.subpage_oversized event includes the sub-page title."""
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        writer = self._make_writer(
            tmp_path,
            responses=[
                self._OVERSIZED_MD,
                self._NORMAL_MD,
                "Chapter index [idx.py:1]",
            ],
        )

        events: list[dict] = []
        _run(writer.generate_chapter(chapter, slices, stream_callback=events.append))

        oversized_events = [e for e in events if e.get("event") == "writer.subpage_oversized"]
        assert len(oversized_events) >= 1
        evt = oversized_events[0]
        assert "title" in evt

    def test_normal_subpage_does_not_emit_oversized_event(self, tmp_path):
        """Sub-pages within 250 lines do NOT trigger writer.subpage_oversized."""
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        # Both sub-pages are short
        writer = self._make_writer(
            tmp_path,
            responses=[
                "Short sp1 [a.py:1].",
                "Short sp2 [b.py:1].",
                "Chapter index [idx.py:1]",
            ],
        )

        events: list[dict] = []
        _run(writer.generate_chapter(chapter, slices, stream_callback=events.append))

        oversized_events = [e for e in events if e.get("event") == "writer.subpage_oversized"]
        assert oversized_events == []

    def test_oversized_content_is_kept_not_split(self, tmp_path):
        """Oversized sub-pages are kept as-is; soft cap never splits content."""
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        writer = self._make_writer(
            tmp_path,
            responses=[
                self._OVERSIZED_MD,
                self._NORMAL_MD,
                "Chapter index [idx.py:1]",
            ],
        )

        _, subpage_mds = _run(writer.generate_chapter(chapter, slices))

        # First sub-page is kept in full (>250 lines)
        assert len(subpage_mds[0].splitlines()) > 250


# ── Tests: fallback on sub-page failure ──────────────────────────────────────


class TestSubPageFallback:
    """When one sub-page fails, its markdown falls back to '# Title\\n\\nDescription\\n'."""

    def test_failing_subpage_uses_fallback_markdown(self, tmp_path):
        """If generate_page raises for a sub-page, fallback is returned for that index."""
        chapter = _make_chapter_spec(n_subpages=2)
        slices = _make_evidence_slices(n=2)

        normal_md = "Normal SP2 content [src/b.py:1]."
        call_count = 0

        def sync_patched_generate_page(spec, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated generation failure")
            return normal_md, []

        # Use a sequential LLM: first two calls are the sub-page attempts (irrelevant
        # because generate_page is patched), then one call for the chapter index.
        llm = SequentialFakeLLM(responses=["Chapter index content [idx.py:1]."])
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=llm)

        with patch.object(writer, "generate_page", side_effect=sync_patched_generate_page):
            index_md, subpage_mds = _run(writer.generate_chapter(chapter, slices))

        # The failing sub-page should have fallback content
        fallback_sp1 = f"# SubPage 1\n\nDescription for sub-page 1.\n"
        assert subpage_mds[0] == fallback_sp1

        # The successful sub-page retains its content
        assert subpage_mds[1] == normal_md
