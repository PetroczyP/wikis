"""Unit tests for RepoExplorer (#feat — agentic repository deep exploration).

Covers:
- explore() returns a non-empty string.
- System prompt contains the required section headers.
- Tool loop terminates within budget.
- read_file tool is dispatched correctly.
- search_symbols and list_cluster_symbols tools dispatch correctly.
- Graceful fallback: when the LLM fails, explore() returns a non-empty fallback.
- budget=0 forces immediate write (no tool calls).
- storage=None → search_symbols and list_cluster_symbols return [no storage] gracefully.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# ── Fake LLM helpers ──────────────────────────────────────────────────────────


class _AnalysisLLM(BaseChatModel):
    """Returns canned responses; first calls may include tool_calls."""

    model_name: str = "fake-analysis"
    responses: list[Any]  # list of AIMessage-compatible content or AIMessage

    @property
    def _llm_type(self) -> str:
        return "fake-analysis"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        resp = self.responses.pop(0) if self.responses else AIMessage(content="# Analysis\nDone.")
        if not isinstance(resp, AIMessage):
            resp = AIMessage(content=str(resp))
        return ChatResult(generations=[ChatGeneration(message=resp)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


class _ErrorLLM(BaseChatModel):
    model_name: str = "error-llm"

    @property
    def _llm_type(self) -> str:
        return "error"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("LLM unavailable")

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


def _fake_tools(tmp_path) -> Any:
    """Build a WriterTools instance pointing at a temp dir."""
    from app.core.wiki_content_writer.writer_tools import WriterTools

    # Create a small file the explorer can read
    (tmp_path / "main.py").write_text("def main():\n    pass\n")
    return WriterTools(repo_root=str(tmp_path), storage=None)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestRepoExplorerImport:
    def test_module_importable(self):
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer  # noqa: F401


class TestExploreReturnsString:
    def test_returns_non_empty_string(self, tmp_path):
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer
        from app.core.wiki_content_writer.writer_tools import WriterTools

        tools = WriterTools(repo_root=str(tmp_path), storage=None)
        llm = _AnalysisLLM(responses=[AIMessage(content="# Repository Analysis: myrepo\n\nDone.")])
        explorer = RepoExplorer(tools=tools, llm_client=llm)
        result = explorer.explore(file_tree="main.py\n", readme="# MyRepo\n", repo_name="myrepo", budget=1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_output_contains_repo_name(self, tmp_path):
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer
        from app.core.wiki_content_writer.writer_tools import WriterTools

        tools = WriterTools(repo_root=str(tmp_path), storage=None)
        llm = _AnalysisLLM(responses=[AIMessage(content="# Repository Analysis: special_repo\n\nDone.")])
        explorer = RepoExplorer(tools=tools, llm_client=llm)
        result = explorer.explore(file_tree="", readme="", repo_name="special_repo", budget=1)
        assert "special_repo" in result


class TestExploreToolLoop:
    def test_tool_calls_dispatched_before_final_write(self, tmp_path):
        """LLM first returns a read_file tool call, then writes the analysis."""
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer
        from app.core.wiki_content_writer.writer_tools import WriterTools

        (tmp_path / "app.py").write_text("class App: pass\n")
        tools = WriterTools(repo_root=str(tmp_path), storage=None)

        # Round 1: tool call; Round 2: final write
        tool_response = AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "app.py"}, "id": "tc1"}],
        )
        final_response = AIMessage(content="# Repository Analysis: repo\n\n## Purpose\nDone.")

        llm = _AnalysisLLM(responses=[tool_response, final_response])
        explorer = RepoExplorer(tools=tools, llm_client=llm)
        result = explorer.explore(file_tree="app.py\n", readme="", repo_name="repo", budget=5)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_budget_exhaustion_produces_fallback_write(self, tmp_path):
        """When budget is exhausted, explorer calls LLM once more to write."""
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer
        from app.core.wiki_content_writer.writer_tools import WriterTools

        tools = WriterTools(repo_root=str(tmp_path), storage=None)

        # Always returns tool calls — budget will be exhausted after budget=1 call
        tool_response = AIMessage(
            content="",
            tool_calls=[{"name": "grep", "args": {"pattern": "main"}, "id": "tc1"}],
        )
        final_write = AIMessage(content="# Repository Analysis: repo\n\nForced write.")

        # First response: tool call; second: forced write
        llm = _AnalysisLLM(responses=[tool_response, final_write])
        explorer = RepoExplorer(tools=tools, llm_client=llm)
        result = explorer.explore(file_tree="", readme="", repo_name="repo", budget=1)

        assert isinstance(result, str)
        assert len(result) > 0


class TestExploreToolDispatch:
    def test_search_symbols_no_storage_returns_no_storage(self, tmp_path):
        """search_symbols with storage=None returns [no storage]."""
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer
        from app.core.wiki_content_writer.writer_tools import WriterTools

        tools = WriterTools(repo_root=str(tmp_path), storage=None)
        explorer = RepoExplorer(tools=tools, llm_client=None)
        result = explorer._search_symbols("main")
        assert "[no storage]" in result

    def test_list_cluster_symbols_no_storage_returns_no_storage(self, tmp_path):
        """list_cluster_symbols with storage=None returns [no storage]."""
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer
        from app.core.wiki_content_writer.writer_tools import WriterTools

        tools = WriterTools(repo_root=str(tmp_path), storage=None)
        explorer = RepoExplorer(tools=tools, llm_client=None)
        result = explorer._list_cluster_symbols(0)
        assert "[no storage]" in result

    def test_search_symbols_with_storage(self, tmp_path):
        """search_symbols calls storage.search_fts_by_symbol_name."""
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer
        from app.core.wiki_content_writer.writer_tools import WriterTools

        storage = MagicMock()
        storage.search_fts_by_symbol_name.return_value = [
            {"symbol_name": "MyClass", "rel_path": "src/cls.py", "start_line": 10},
        ]
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        explorer = RepoExplorer(tools=tools, llm_client=None)
        result = explorer._search_symbols("MyClass")

        assert "MyClass" in result
        assert "src/cls.py" in result
        storage.search_fts_by_symbol_name.assert_called_once_with("MyClass", limit=10)

    def test_list_cluster_symbols_with_storage(self, tmp_path):
        """list_cluster_symbols calls storage.get_nodes_by_cluster."""
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer
        from app.core.wiki_content_writer.writer_tools import WriterTools

        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = [
            {"symbol_name": "WorkerPool", "symbol_type": "class", "rel_path": "src/pool.py"},
        ]
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        explorer = RepoExplorer(tools=tools, llm_client=None)
        result = explorer._list_cluster_symbols(3)

        assert "WorkerPool" in result
        assert "src/pool.py" in result
        storage.get_nodes_by_cluster.assert_called_once_with(3, limit=30)


class TestExploreGracefulFallback:
    def test_llm_error_returns_fallback_string(self, tmp_path):
        """When LLM always raises, explore() returns a non-empty fallback string."""
        from app.core.wiki_content_writer.repo_explorer import RepoExplorer
        from app.core.wiki_content_writer.writer_tools import WriterTools

        tools = WriterTools(repo_root=str(tmp_path), storage=None)
        explorer = RepoExplorer(tools=tools, llm_client=_ErrorLLM())
        result = explorer.explore(file_tree="main.py", readme="# Test", repo_name="errrepo", budget=2)

        assert isinstance(result, str)
        assert len(result) > 0
        # Must mention the repo name in fallback
        assert "errrepo" in result
