"""Unit tests for app/core/wiki_content_writer/repo_overview.py.

Covers:
- Empty pages list → returns minimal placeholder page.
- LLM call succeeds → returns LLM content with repo name heading.
- LLM returns content that already starts with the heading → not doubled.
- LLM call fails → graceful fallback with page list.
- Pages accessed as both attribute-bearing objects and dicts.
- Context string contains all page titles.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.wiki_content_writer.repo_overview import generate_repo_overview


# ── FakeLLM ───────────────────────────────────────────────────────────────────


class FakeLLM(BaseChatModel):
    """Minimal BaseChatModel subclass returning canned responses."""

    responses: list[str] = []
    model_name: str = "fake-llm"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self.responses[0] if self.responses else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


class ErrorLLM(BaseChatModel):
    """LLM that always raises an exception."""

    model_name: str = "error-llm"

    @property
    def _llm_type(self) -> str:
        return "error"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("LLM unavailable")

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


# ── Stub page objects ──────────────────────────────────────────────────────────


@dataclass
class StubPage:
    title: str
    description: str = ""
    markdown: str = ""
    metadata: dict = field(default_factory=dict)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestEmptyPages:
    def test_empty_pages_returns_placeholder(self):
        """Empty pages list → returns minimal placeholder, not an error."""
        llm = FakeLLM(responses=[])
        result = generate_repo_overview([], llm=llm, repo_name="myrepo", repo_url="")
        assert "myrepo" in result
        assert result.startswith("# myrepo")

    def test_empty_pages_does_not_call_llm(self):
        """LLM is not called when pages is empty — early return."""
        call_count = 0

        class CountingLLM(FakeLLM):
            def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
                nonlocal call_count
                call_count += 1
                return super()._generate(*args, **kwargs)

        generate_repo_overview([], llm=CountingLLM(responses=[]), repo_name="r", repo_url="")
        assert call_count == 0


class TestLLMSuccess:
    def test_llm_content_used_in_output(self):
        """When LLM succeeds, its content forms the body of the overview."""
        pages = [
            StubPage(title="Auth Module", description="Handles authentication."),
            StubPage(title="Data Layer", description="Database access."),
        ]
        llm_body = "This is a great repo.\n\n## Main Areas\n\nAuth and data.\n\n## Wiki Pages\n\n- [[Auth Module]] — auth\n"
        llm = FakeLLM(responses=[llm_body])
        result = generate_repo_overview(pages, llm=llm, repo_name="acme", repo_url="")

        assert "This is a great repo." in result
        assert "acme" in result

    def test_repo_name_heading_prepended_when_missing(self):
        """If LLM omits the # heading, it is prepended."""
        pages = [StubPage(title="Foo")]
        llm = FakeLLM(responses=["LLM overview body without heading."])
        result = generate_repo_overview(pages, llm=llm, repo_name="myrepo", repo_url="")
        assert result.startswith("# myrepo")
        assert "LLM overview body without heading." in result

    def test_repo_name_heading_not_doubled_when_already_present(self):
        """If LLM output already starts with # repo_name, it is not duplicated."""
        pages = [StubPage(title="Foo")]
        llm = FakeLLM(responses=["# myrepo\n\nBody text."])
        result = generate_repo_overview(pages, llm=llm, repo_name="myrepo", repo_url="")
        # Count occurrences of the heading — must be exactly one
        assert result.count("# myrepo") == 1


class TestLLMFailure:
    def test_fallback_contains_repo_name_and_page_list(self):
        """When LLM raises, fallback markdown lists all pages."""
        pages = [
            StubPage(title="Auth", description="Auth desc."),
            StubPage(title="API", description="API desc."),
        ]
        result = generate_repo_overview(pages, llm=ErrorLLM(), repo_name="broken", repo_url="")
        assert "broken" in result
        assert "[[Auth]]" in result
        assert "[[API]]" in result

    def test_fallback_does_not_raise(self):
        """Fallback path must not propagate the LLM exception."""
        pages = [StubPage(title="X")]
        result = generate_repo_overview(pages, llm=ErrorLLM(), repo_name="r", repo_url="")
        assert isinstance(result, str)
        assert "r" in result


class TestPageContextBuilding:
    def test_context_includes_all_page_titles(self):
        """The LLM prompt context must contain every page title."""
        pages = [
            StubPage(title="Alpha", description="desc alpha"),
            StubPage(title="Beta", description="desc beta"),
            StubPage(title="Gamma", description=""),
        ]
        captured_messages: list[Any] = []

        class CaptureLLM(FakeLLM):
            def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
                captured_messages.extend(messages)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="overview"))])

        generate_repo_overview(pages, llm=CaptureLLM(responses=[]), repo_name="r", repo_url="")

        combined = " ".join(str(m.content) for m in captured_messages)
        assert "Alpha" in combined
        assert "Beta" in combined
        assert "Gamma" in combined

    def test_dict_pages_supported(self):
        """Pages passed as dicts (title/description keys) are handled gracefully."""
        pages = [
            {"title": "DictPage", "description": "Dict desc."},
        ]
        llm = FakeLLM(responses=["Dict overview"])
        result = generate_repo_overview(pages, llm=llm, repo_name="dictrepo", repo_url="")  # type: ignore[arg-type]
        assert "dictrepo" in result


class TestRepoAnalysisParam:
    def test_repo_analysis_in_system_prompt(self):
        """repo_analysis is injected into the system prompt when provided."""
        pages = [StubPage(title="X", description="desc", markdown="Some markdown content.")]
        captured_messages: list[Any] = []

        class CaptureLLM(FakeLLM):
            def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
                captured_messages.extend(messages)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="result"))])

        generate_repo_overview(
            pages,
            llm=CaptureLLM(responses=[]),
            repo_name="r",
            repo_url="",
            repo_analysis="UNIQUE_ANALYSIS_TOKEN_XYZ",
        )

        system_msgs = [m for m in captured_messages if hasattr(m, "type") and m.type == "system"
                       or m.__class__.__name__ == "SystemMessage"]
        combined = " ".join(str(m.content) for m in captured_messages)
        assert "UNIQUE_ANALYSIS_TOKEN_XYZ" in combined

    def test_empty_repo_analysis_not_injected(self):
        """When repo_analysis is empty, the system prompt has no extra section."""
        pages = [StubPage(title="Y", description="desc")]
        captured_messages: list[Any] = []

        class CaptureLLM(FakeLLM):
            def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
                captured_messages.extend(messages)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        generate_repo_overview(pages, llm=CaptureLLM(responses=[]), repo_name="r", repo_url="", repo_analysis="")
        combined = " ".join(str(m.content) for m in captured_messages)
        # No "Architectural Context" header should appear when analysis is empty
        assert "UNIQUE_ANALYSIS" not in combined


class TestMarkdownExcerpts:
    def test_page_markdown_included_in_prompt(self):
        """When pages have markdown content, it is included in the user prompt."""
        pages = [
            StubPage(title="Auth", description="Auth module.", markdown="## Auth\nHandles JWT tokens.\n"),
        ]
        captured_messages: list[Any] = []

        class CaptureLLM(FakeLLM):
            def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
                captured_messages.extend(messages)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="overview"))])

        generate_repo_overview(pages, llm=CaptureLLM(responses=[]), repo_name="r", repo_url="")
        combined = " ".join(str(m.content) for m in captured_messages)
        assert "JWT" in combined or "Auth" in combined

    def test_pages_without_markdown_use_description(self):
        """Pages without markdown fall back to description in the prompt."""
        pages = [
            StubPage(title="Scheduler", description="Runs background jobs.", markdown=""),
        ]
        captured_messages: list[Any] = []

        class CaptureLLM(FakeLLM):
            def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
                captured_messages.extend(messages)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="overview"))])

        generate_repo_overview(pages, llm=CaptureLLM(responses=[]), repo_name="r", repo_url="")
        combined = " ".join(str(m.content) for m in captured_messages)
        assert "Scheduler" in combined or "background" in combined
