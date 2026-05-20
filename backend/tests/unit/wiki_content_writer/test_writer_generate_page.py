"""Unit tests for WikiContentWriter.generate_page (#285).

Covers:
- generate_page returns (markdown, tool_trace) with known FakeLLM output.
- tool_trace is populated when the LLM requests tool calls.
- zero tool-call path (LLM writes on first turn) works correctly.
- stream_callback receives writer.tool_call events.
- citation format written by the writer is parseable by extract_citations.
- zero-citation path still returns content (not empty string).
- missing llm raises ValueError.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall as LCToolCall
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.wiki_content_writer.citation_extractor import extract_citations
from app.core.wiki_content_writer.source_gate import ToolCall
from app.core.wiki_content_writer.writer_agent import WikiContentWriter
from app.core.wiki_content_writer.writer_tools import WriterTools


# ── FakeLLM helpers ──────────────────────────────────────────────────────────


class _BindToolsNoop:
    """Mixin: bind_tools returns self so writer_agent's direct call succeeds."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self


class DirectWriterFakeLLM(_BindToolsNoop, BaseChatModel):
    """FakeLLM that returns a canned markdown string on the first call (no tools)."""

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


class ToolThenWriteFakeLLM(_BindToolsNoop, BaseChatModel):
    """FakeLLM: first call returns a tool-call request; second returns markdown."""

    tool_name: str = "read_file"
    tool_args: dict = {"path": "src/auth.py"}
    writer_response: str = ""
    call_count: int = 0
    model_name: str = "tool-then-write-fake"

    @property
    def _llm_type(self) -> str:
        return "tool-then-write-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        if self.call_count == 1:
            # Emit a tool call on the first turn
            lc_tc = LCToolCall(name=self.tool_name, args=self.tool_args, id="tc1")
            msg = AIMessage(content="")
            msg.tool_calls = [lc_tc]  # type: ignore[attr-defined]
        else:
            # Write the page on the second turn
            msg = AIMessage(content=self.writer_response)
            msg.tool_calls = []  # type: ignore[attr-defined]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestGeneratePageDirectWrite:
    """LLM writes on first call (no tool calls)."""

    _MARKDOWN = (
        "The AuthService validates JWT tokens. [src/auth.py:10-20]\n\n"
        "Session expiry is checked in the middleware. [middleware.py:55]"
    )

    def _make_writer(self, tmp_path):
        llm = DirectWriterFakeLLM(response=self._MARKDOWN)
        tools = WriterTools(repo_root=str(tmp_path))
        return WikiContentWriter(tools=tools, llm_client=llm)

    def _spec(self):
        return {
            "title": "Authentication Module",
            "description": "Handles JWT auth.",
            "retrieval_query": "auth session jwt",
            "target_symbols": [("AuthService", "public_api")],
            "target_folders": ["src/auth"],
            "target_docs": [],
        }

    def test_returns_markdown_string(self, tmp_path):
        writer = self._make_writer(tmp_path)
        md, _ = writer.generate_page(self._spec())
        assert isinstance(md, str)
        assert md == self._MARKDOWN

    def test_returns_empty_tool_trace(self, tmp_path):
        writer = self._make_writer(tmp_path)
        _, trace = writer.generate_page(self._spec())
        assert trace == []

    def test_markdown_citations_parseable(self, tmp_path):
        writer = self._make_writer(tmp_path)
        md, _ = writer.generate_page(self._spec())
        claims = extract_citations(md)
        # Both paragraphs carry citations
        assert len(claims) == 2
        assert all(len(c.citations) > 0 for c in claims)

    def test_accepts_dataclass_spec(self, tmp_path):
        """generate_page accepts a PageSpec dataclass (not just dict)."""
        from dataclasses import dataclass, field

        @dataclass
        class FakePageSpec:
            title: str = "Auth"
            description: str = "Auth module."
            retrieval_query: str = "auth"
            target_symbols: list = field(default_factory=list)
            target_folders: list = field(default_factory=list)
            target_docs: list = field(default_factory=list)
            cluster_id: int = 1

        writer = self._make_writer(tmp_path)
        md, _ = writer.generate_page(FakePageSpec())
        assert md == self._MARKDOWN

    def test_no_stream_callback_does_not_error(self, tmp_path):
        writer = self._make_writer(tmp_path)
        # Should not raise even without stream_callback
        md, trace = writer.generate_page(self._spec(), stream_callback=None)
        assert md

    def test_stream_callback_receives_no_tool_events_on_direct_write(self, tmp_path):
        writer = self._make_writer(tmp_path)
        events: list[dict] = []
        writer.generate_page(self._spec(), stream_callback=events.append)
        tool_events = [e for e in events if e.get("event") == "writer.tool_call"]
        assert tool_events == []


class TestGeneratePageWithToolCalls:
    """LLM requests a tool on first turn, then writes on second."""

    _WRITER_MD = "AuthService validates tokens. [src/auth.py:10-20]"

    def _make_writer(self, tmp_path, content="class AuthService: pass\n"):
        # Create the file so read_file tool succeeds
        src = tmp_path / "src" / "auth.py"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(content)

        llm = ToolThenWriteFakeLLM(
            tool_name="read_file",
            tool_args={"path": "src/auth.py"},
            writer_response=self._WRITER_MD,
        )
        tools = WriterTools(repo_root=str(tmp_path))
        return WikiContentWriter(tools=tools, llm_client=llm)

    def _spec(self):
        return {
            "title": "Auth",
            "description": "Auth module.",
            "retrieval_query": "auth",
            "target_symbols": [("AuthService", "public_api")],
            "target_folders": ["src"],
            "target_docs": [],
        }

    def test_returns_writer_markdown(self, tmp_path):
        writer = self._make_writer(tmp_path)
        md, _ = writer.generate_page(self._spec())
        assert md == self._WRITER_MD

    def test_tool_trace_populated(self, tmp_path):
        writer = self._make_writer(tmp_path)
        _, trace = writer.generate_page(self._spec())
        assert len(trace) == 1
        assert trace[0].tool == "read_file"

    def test_tool_trace_has_result_text(self, tmp_path):
        writer = self._make_writer(tmp_path)
        _, trace = writer.generate_page(self._spec())
        # result_text should include the file content
        assert "AuthService" in trace[0].result_text

    def test_tool_trace_entries_are_ToolCall(self, tmp_path):
        writer = self._make_writer(tmp_path)
        _, trace = writer.generate_page(self._spec())
        for entry in trace:
            assert isinstance(entry, ToolCall)

    def test_stream_callback_receives_tool_call_event(self, tmp_path):
        writer = self._make_writer(tmp_path)
        events: list[dict] = []
        writer.generate_page(self._spec(), stream_callback=events.append)
        tool_events = [e for e in events if e.get("event") == "writer.tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["tool"] == "read_file"


class TestGeneratePageEdgeCases:
    def test_missing_llm_raises_valueerror(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=None)
        with pytest.raises(ValueError, match="no LLM configured"):
            writer.generate_page({"title": "X", "description": "Y"})

    def test_llm_override_takes_precedence_over_client(self, tmp_path):
        """llm= kwarg overrides llm_client."""
        _OVERRIDE_MD = "Override content. [override.py:1]"
        override_llm = DirectWriterFakeLLM(response=_OVERRIDE_MD)
        # Set a different llm_client that would return different content
        default_llm = DirectWriterFakeLLM(response="Default content. [default.py:1]")
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=default_llm)
        md, _ = writer.generate_page(
            {"title": "T", "description": "D"},
            llm=override_llm,
        )
        assert md == _OVERRIDE_MD

    def test_zero_citation_content_still_returned(self, tmp_path):
        """When writer produces no citations, generate_page still returns the content."""
        no_cite_md = "This paragraph has no citation token."
        llm = DirectWriterFakeLLM(response=no_cite_md)
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=llm)
        md, _ = writer.generate_page({"title": "T", "description": "D"})
        assert md == no_cite_md


# ── JSON-string args coercion (Rio Blocking fix) ──────────────────────────────


class JsonStringArgsFakeLLM(_BindToolsNoop, BaseChatModel):
    """FakeLLM that emits a tool_call with args as a JSON *string* (not a dict).

    Some LangChain providers / versions return tool_call.args as a serialised
    JSON string rather than a pre-parsed dict.  The writer loop must coerce
    the string before calling .get(…) on it; otherwise str.get returns None,
    read_file is invoked with path="" and returns "[empty file]".
    """

    tool_name: str = "read_file"
    # args as a JSON string — this is the provider behaviour we are guarding against
    tool_args_str: str = '{"path": "src/auth.py"}'
    writer_response: str = ""
    call_count: int = 0
    model_name: str = "json-string-args-fake"

    @property
    def _llm_type(self) -> str:
        return "json-string-args-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        if self.call_count == 1:
            # Return a tool_call whose args is a *string*, not a dict.
            lc_tc = LCToolCall(
                name=self.tool_name,
                args=self.tool_args_str,  # type: ignore[arg-type]  # deliberate string
                id="tc_json_str",
            )
            msg = AIMessage(content="")
            msg.tool_calls = [lc_tc]  # type: ignore[attr-defined]
        else:
            msg = AIMessage(content=self.writer_response)
            msg.tool_calls = []  # type: ignore[attr-defined]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


class TestJsonStringArgsCoercion:
    """Guard: tool-call args that arrive as a JSON string are correctly dispatched."""

    _WRITER_MD = "AuthService validates tokens. [src/auth.py:10-20]"

    def _make_writer(self, tmp_path):
        src = tmp_path / "src" / "auth.py"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("class AuthService: pass\n")

        llm = JsonStringArgsFakeLLM(
            tool_name="read_file",
            tool_args_str='{"path": "src/auth.py"}',
            writer_response=self._WRITER_MD,
        )
        tools = WriterTools(repo_root=str(tmp_path))
        return WikiContentWriter(tools=tools, llm_client=llm)

    def _spec(self):
        return {
            "title": "Auth",
            "description": "Auth module.",
            "retrieval_query": "auth",
            "target_symbols": [("AuthService", "public_api")],
            "target_folders": ["src"],
            "target_docs": [],
        }

    def test_json_string_args_dispatched_correctly(self, tmp_path):
        """JSON-string args are coerced to dict before dispatch — read_file gets the right path."""
        writer = self._make_writer(tmp_path)
        md, trace = writer.generate_page(self._spec())
        # The tool call must have been dispatched (trace is non-empty)
        assert len(trace) == 1, "Tool call was not dispatched"
        assert trace[0].tool == "read_file"
        # The result must contain the actual file content — not "[empty file]"
        assert "[empty file]" not in trace[0].result_text, (
            "read_file returned '[empty file]' — args were not coerced from JSON string to dict"
        )
        assert "AuthService" in trace[0].result_text

    def test_json_string_args_writer_gets_content(self, tmp_path):
        """When args are coerced correctly the writer produces non-empty markdown."""
        writer = self._make_writer(tmp_path)
        md, _ = writer.generate_page(self._spec())
        assert md == self._WRITER_MD


# ── Multi-tool-call budget capping ────────────────────────────────────────────


class MultiToolCallFakeLLM(_BindToolsNoop, BaseChatModel):
    """FakeLLM that emits N tool calls in a single response, then writes.

    Used to verify that the budget is counted per tool invocation, not per
    LLM round — so a single multi-tool response cannot exceed the cap.
    """

    n_tools: int = 3
    tool_name: str = "grep"
    writer_response: str = ""
    call_count: int = 0
    model_name: str = "multi-tool-fake"

    @property
    def _llm_type(self) -> str:
        return "multi-tool-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        if self.call_count == 1:
            # Emit n_tools tool calls in one response.
            tool_calls = [
                LCToolCall(name=self.tool_name, args={"pattern": f"query_{i}"}, id=f"tc_{i}")
                for i in range(self.n_tools)
            ]
            msg = AIMessage(content="")
            msg.tool_calls = tool_calls  # type: ignore[attr-defined]
        else:
            msg = AIMessage(content=self.writer_response)
            msg.tool_calls = []  # type: ignore[attr-defined]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


class TestMultiToolCallBudgetCapping:
    """Budget is enforced on total tool invocations, not LLM rounds."""

    _WRITER_MD = "Grep results cited. [src/auth.py:1]"

    def test_budget_caps_total_tool_invocations(self, tmp_path):
        """Budget is a soft cap: all tool_calls in the CURRENT response are
        processed before the budget is checked.  A response with N > budget
        tool calls is fully executed so Bedrock's message-structure contract
        is upheld (every tool_use must have a tool_result immediately after).
        Subsequent LLM rounds are suppressed, so the total can exceed budget
        by at most N-1 within one response."""
        tools = WriterTools(repo_root=str(tmp_path))
        # LLM emits 5 tool calls in one round, budget is 3.
        llm = MultiToolCallFakeLLM(n_tools=5, writer_response=self._WRITER_MD)
        writer = WikiContentWriter(tools=tools, llm_client=llm, max_iterations=3)
        _, trace = writer.generate_page(
            {"title": "T", "description": "D", "retrieval_query": "q",
             "target_symbols": [], "target_folders": [], "target_docs": []}
        )
        # All 5 calls in the single response are processed (soft cap).
        # No further rounds run, so total == 5, not > 5+budget.
        assert len(trace) == 5, (
            f"Expected exactly 5 (one full response), got {len(trace)}"
        )

    def test_budget_not_exceeded_across_rounds(self, tmp_path):
        """Total tool calls across all rounds never exceeds budget."""
        tools = WriterTools(repo_root=str(tmp_path))
        llm = MultiToolCallFakeLLM(n_tools=2, writer_response=self._WRITER_MD)
        budget = 5
        writer = WikiContentWriter(tools=tools, llm_client=llm, max_iterations=budget)
        _, trace = writer.generate_page(
            {"title": "T", "description": "D", "retrieval_query": "q",
             "target_symbols": [], "target_folders": [], "target_docs": []}
        )
        assert len(trace) <= budget


# ── Citation format regression: writer format matches extractor regex ─────────


class TestCitationFormatRoundtrip:
    """Verify that the citation format described in prompts.py is parseable by extract_citations."""

    def test_single_line_format(self):
        md = "The function validates the token [api/auth.py:42]."
        claims = extract_citations(md)
        assert len(claims) == 1
        assert claims[0].citations[0].path == "api/auth.py"
        assert claims[0].citations[0].start_line == 42
        assert claims[0].citations[0].end_line == 42

    def test_range_format(self):
        md = "Session expiry logic [middleware/session.py:10-30]."
        claims = extract_citations(md)
        cite = claims[0].citations[0]
        assert cite.path == "middleware/session.py"
        assert cite.start_line == 10
        assert cite.end_line == 30

    def test_multi_file_comma_separated(self):
        md = "Config loaded from env [config.py:5-12] and merged [config.py:40-48]."
        claims = extract_citations(md)
        assert len(claims[0].citations) == 2

    def test_multiple_paragraphs_all_cited(self):
        md = (
            "AuthService validates tokens [src/auth.py:10-20].\n\n"
            "Session expiry is checked in middleware. [middleware.py:55]"
        )
        claims = extract_citations(md)
        assert len(claims) == 2
        assert all(len(c.citations) > 0 for c in claims)

    def test_uncited_paragraph_has_empty_citations(self):
        md = "This paragraph has no citation."
        claims = extract_citations(md)
        assert len(claims) == 1
        assert claims[0].citations == []


# ── Regression: bind_tools called directly (no try/except fallback) ──────────


class TestWriterBindToolsDirect:
    """Regression for #295: writer must call bind_tools without a fallback path.

    Before the fix, generate_page had a try/except that silently disabled tools
    when bind_tools raised AttributeError (RunnableRetry wrapper).  The fix
    removes that fallback; the LLM from create_llm() always exposes bind_tools.
    """

    class BindToolsRecordingLLM(_BindToolsNoop, BaseChatModel):
        """FakeLLM that records whether bind_tools was called and with which schemas."""

        bind_tools_called: bool = False
        bind_tools_schemas: list = []
        response: str = "Done writing. [src/app.py:1]"
        model_name: str = "bind-tools-recording"

        @property
        def _llm_type(self) -> str:
            return "bind-tools-recording"

        def bind_tools(self, tools: Any, **kwargs: Any) -> "TestWriterBindToolsDirect.BindToolsRecordingLLM":
            self.bind_tools_called = True
            self.bind_tools_schemas = list(tools)
            return self

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

    def test_writer_calls_bind_tools_on_llm(self, tmp_path):
        """generate_page calls effective_llm.bind_tools() unconditionally."""
        llm = self.BindToolsRecordingLLM()
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=llm)
        writer.generate_page({"title": "T", "description": "D"})
        assert llm.bind_tools_called, (
            "writer did not call bind_tools — silent tool-less fallback path may be active"
        )

    def test_writer_binds_expected_tool_schemas(self, tmp_path):
        """bind_tools is called with the six writer tool schemas."""
        llm = self.BindToolsRecordingLLM()
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, llm_client=llm)
        writer.generate_page({"title": "T", "description": "D"})

        tool_names = {s["function"]["name"] for s in llm.bind_tools_schemas}
        assert tool_names == {
            "read_file", "get_signature", "grep",
            "get_callers", "get_callees", "list_doc_chunks",
        }
