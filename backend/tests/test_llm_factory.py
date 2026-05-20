"""Tests for LLM/embedding factory."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.services.llm_factory import create_embeddings, create_llm, safe_bind_tools

_has_ollama = importlib.util.find_spec("langchain_ollama") is not None
_has_bedrock = importlib.util.find_spec("langchain_aws") is not None


def _settings(**overrides) -> Settings:
    defaults = {
        "llm_provider": "openai",
        "llm_api_key": SecretStr("test-key"),
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
        "llm_max_retries": 0,  # disable retry wrapper in tests
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestCreateLLM:
    def test_openai_provider(self):
        llm = create_llm(_settings())
        assert type(llm).__name__ == "ChatOpenAI"
        assert llm.model_name == "gpt-4o-mini"

    def test_anthropic_provider(self):
        llm = create_llm(_settings(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514"))
        assert type(llm).__name__ == "ChatAnthropic"

    def test_custom_provider_uses_openai_with_base_url(self):
        s = _settings(llm_provider="custom", llm_api_base="http://localhost:11434/v1")
        llm = create_llm(s)
        assert type(llm).__name__ == "ChatOpenAI"

    def test_temperature_heuristic_reasoning_model(self):
        llm = create_llm(_settings(llm_model="o1-preview"))
        assert llm.temperature == 1.0

    def test_temperature_heuristic_standard_model(self):
        llm = create_llm(_settings(llm_model="gpt-4o-mini"))
        assert llm.temperature == 0.1

    def test_temperature_heuristic_namespaced_model(self):
        """Namespaced models like openai/gpt-4o should not trigger reasoning temp."""
        llm = create_llm(_settings(llm_model="openai/gpt-4o"))
        assert llm.temperature == 0.1

    def test_override_kwargs(self):
        llm = create_llm(_settings(), temperature=0.5, max_tokens=1024)
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm(_settings(llm_provider="unknown"))

    @pytest.mark.skipif(not _has_ollama, reason="langchain-ollama not installed")
    def test_ollama_provider(self):
        llm = create_llm(_settings(llm_provider="ollama", llm_model="llama3"))
        assert type(llm).__name__ == "ChatOllama"

    def test_gemini_provider(self):
        s = _settings(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key=SecretStr("gk"))
        llm = create_llm(s)
        assert type(llm).__name__ == "ChatGoogleGenerativeAI"

    def test_gemini_missing_key_raises(self):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            create_llm(_settings(llm_provider="gemini", gemini_api_key=None))

    def test_copilot_provider(self):
        llm = create_llm(_settings(llm_provider="copilot", llm_model="gpt-4o"))
        assert type(llm).__name__ == "ChatOpenAI"
        assert llm.openai_api_base == "https://api.githubcopilot.com"

    def test_github_provider_openai_model(self):
        llm = create_llm(_settings(llm_provider="github", llm_model="openai/gpt-4o"))
        assert type(llm).__name__ == "ChatOpenAI"
        assert llm.openai_api_base == "https://models.inference.ai.azure.com"

    def test_github_provider_claude_model(self):
        llm = create_llm(_settings(llm_provider="github", llm_model="claude-sonnet-4-5"))
        assert type(llm).__name__ == "ChatAnthropic"

    def test_github_provider_anthropic_prefixed_model(self):
        llm = create_llm(_settings(llm_provider="github", llm_model="anthropic/claude-sonnet-4-5"))
        assert type(llm).__name__ == "ChatAnthropic"

    @pytest.mark.skipif(not _has_bedrock, reason="langchain-aws not installed")
    def test_bedrock_provider(self):
        llm = create_llm(_settings(llm_provider="bedrock", llm_model="anthropic.claude-3-sonnet"))
        assert type(llm).__name__ == "ChatBedrock"


class TestCreateEmbeddings:
    def test_openai_embeddings_with_key(self):
        emb = create_embeddings(_settings())
        assert type(emb).__name__ == "OpenAIEmbeddings"

    def test_falls_back_to_llm_key(self):
        s = _settings(embedding_api_key=None)
        emb = create_embeddings(s)
        assert type(emb).__name__ == "OpenAIEmbeddings"

    def test_uses_dedicated_embedding_key(self):
        s = _settings(embedding_api_key=SecretStr("embed-key"))
        emb = create_embeddings(s)
        assert type(emb).__name__ == "OpenAIEmbeddings"

    def test_no_key_raises(self):
        s = _settings(llm_api_key=SecretStr(""), embedding_api_key=None)
        with pytest.raises(ValueError, match="No API key"):
            create_embeddings(s)

    def test_model_override(self):
        emb = create_embeddings(_settings(), model="text-embedding-ada-002")
        assert emb.model == "text-embedding-ada-002"

    @pytest.mark.skipif(not _has_ollama, reason="langchain-ollama not installed")
    def test_ollama_embeddings(self):
        emb = create_embeddings(_settings(llm_provider="ollama"))
        assert type(emb).__name__ == "OllamaEmbeddings"

    def test_gemini_embeddings(self):
        emb = create_embeddings(_settings(llm_provider="gemini", gemini_api_key=SecretStr("gk")))
        assert type(emb).__name__ == "GoogleGenerativeAIEmbeddings"

    def test_gemini_embeddings_missing_key_raises(self):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            create_embeddings(_settings(llm_provider="gemini", gemini_api_key=None))

    @pytest.mark.skipif(not _has_bedrock, reason="langchain-aws not installed")
    def test_bedrock_embeddings(self):
        emb = create_embeddings(_settings(llm_provider="bedrock"))
        assert type(emb).__name__ == "BedrockEmbeddings"

    def test_anthropic_raises_without_embedding_key(self):
        with pytest.raises(ValueError, match="no embedding API"):
            create_embeddings(_settings(llm_provider="anthropic", embedding_api_key=None))

    def test_anthropic_falls_back_to_openai_with_embedding_key(self):
        emb = create_embeddings(_settings(llm_provider="anthropic", embedding_api_key=SecretStr("emb-key")))
        assert type(emb).__name__ == "OpenAIEmbeddings"

    def test_copilot_raises_without_embedding_key(self):
        with pytest.raises(ValueError, match="no embedding API"):
            create_embeddings(_settings(llm_provider="copilot", embedding_api_key=None))

    def test_copilot_falls_back_to_openai_with_embedding_key(self):
        emb = create_embeddings(_settings(llm_provider="copilot", embedding_api_key=SecretStr("emb-key")))
        assert type(emb).__name__ == "OpenAIEmbeddings"

    def test_embedding_provider_override(self):
        """EMBEDDING_PROVIDER overrides LLM_PROVIDER for embeddings."""
        emb = create_embeddings(_settings(llm_provider="anthropic", embedding_provider="openai", embedding_api_key=SecretStr("emb-key")))
        assert type(emb).__name__ == "OpenAIEmbeddings"

    def test_github_embeddings(self):
        emb = create_embeddings(_settings(llm_provider="github"))
        assert type(emb).__name__ == "OpenAIEmbeddings"


# ── bind_tools-before-retry tests ────────────────────────────────────────────


class TestCreateLLMBindTools:
    """Verify tools= is bound on the base LLM before retry/fallback wrappers."""

    _FAKE_TOOL = {
        "type": "function",
        "function": {
            "name": "fake_tool",
            "description": "A fake tool for testing.",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        },
    }

    def test_tools_bound_without_retry(self):
        """With max_retries=0 (no wrapper) the result is a _ChatModelBinding with tools in .kwargs."""
        llm = create_llm(_settings(llm_max_retries=0), tools=[self._FAKE_TOOL])
        # bind_tools returns a _ChatModelBinding; tools are stored in .kwargs
        kwargs = getattr(llm, "kwargs", {})
        assert "tools" in kwargs, "bind_tools was not applied — 'tools' missing from .kwargs"
        assert kwargs["tools"] == [self._FAKE_TOOL]

    def test_tools_bound_with_retry_wrapper(self):
        """With max_retries>0 tools must survive the retry wrap.

        LangChain's bind_tools returns _ChatModelBinding then .with_retry() wraps
        it: _ChatModelBinding (tools in .kwargs) -> RunnableRetry.
        The outermost object still carries .kwargs["tools"].
        """
        llm = create_llm(_settings(llm_max_retries=2), tools=[self._FAKE_TOOL])
        # bind_tools is called BEFORE with_retry, so the outermost wrapper
        # is _ChatModelBinding and its .kwargs["tools"] must be present.
        kwargs = getattr(llm, "kwargs", {})
        assert "tools" in kwargs, (
            f"tools not found on outer wrapper ({type(llm).__name__}); "
            "bind_tools may have been called after with_retry"
        )
        assert kwargs["tools"] == [self._FAKE_TOOL]

    def test_no_tools_arg_omits_tool_binding(self):
        """When tools= is not passed the LLM is returned without any tool binding."""
        llm = create_llm(_settings(llm_max_retries=0))
        kwargs = getattr(llm, "kwargs", {})
        assert "tools" not in kwargs

    def test_empty_tools_list_omits_tool_binding(self):
        """An empty tools list should not call bind_tools at all."""
        llm = create_llm(_settings(llm_max_retries=0), tools=[])
        kwargs = getattr(llm, "kwargs", {})
        assert "tools" not in kwargs


# ── safe_bind_tools tests ─────────────────────────────────────────────────────


class TestSafeBindTools:
    """safe_bind_tools unwraps retry/fallback wrappers and binds at the base."""

    _TOOLS = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}]

    # ── helpers ──────────────────────────────────────────────────────────────

    class _BindCaptureLLM:
        """Minimal stand-in that captures bind_tools calls."""

        def __init__(self) -> None:
            self.bound_tools: list[Any] | None = None

        def bind_tools(self, tools: list[Any], **kwargs: Any) -> "_BindCaptureLLM":
            self.bound_tools = list(tools)
            return self

    class _NoBindLLM:
        """Stand-in without bind_tools — simulates a broken/opaque wrapper."""
        pass

    class _FakeRetry:
        """Simulates RunnableRetry: exposes .bound pointing to inner LLM."""

        def __init__(self, inner: Any) -> None:
            self.bound = inner

    class _FakeFallbacks:
        """Simulates RunnableWithFallbacks: exposes .runnable pointing to inner."""

        def __init__(self, inner: Any) -> None:
            self.runnable = inner

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_direct_bind_on_unwrapped_llm(self) -> None:
        """If the LLM has bind_tools directly, it is called without unwrapping."""
        llm = self._BindCaptureLLM()
        result = safe_bind_tools(llm, self._TOOLS)
        assert result is llm, "safe_bind_tools must return the bind_tools result"
        assert llm.bound_tools == self._TOOLS

    def test_bind_through_retry_wrapper(self) -> None:
        """When outermost object is RunnableRetry-like (.bound chain), inner LLM is found."""
        inner = self._BindCaptureLLM()
        retry_wrap = self._FakeRetry(inner)
        result = safe_bind_tools(retry_wrap, self._TOOLS)
        assert result is inner, "safe_bind_tools must return the result of inner.bind_tools()"
        assert inner.bound_tools == self._TOOLS

    def test_bind_through_double_wrap(self) -> None:
        """Retry → Fallbacks → Base: walks both .bound and .runnable chains."""
        inner = self._BindCaptureLLM()
        fallbacks_wrap = self._FakeFallbacks(inner)
        retry_wrap = self._FakeRetry(fallbacks_wrap)
        result = safe_bind_tools(retry_wrap, self._TOOLS)
        assert result is inner
        assert inner.bound_tools == self._TOOLS

    def test_raises_when_no_bind_tools_found(self) -> None:
        """Raises TypeError with a clear message when no inner LLM supports bind_tools."""
        opaque = self._NoBindLLM()
        with pytest.raises(TypeError, match="bind_tools"):
            safe_bind_tools(opaque, self._TOOLS)

    def test_bind_through_runnable_fallbacks_wrapper(self) -> None:
        """If outermost uses .runnable (RunnableWithFallbacks-like), inner is found."""
        inner = self._BindCaptureLLM()
        fallbacks_wrap = self._FakeFallbacks(inner)
        result = safe_bind_tools(fallbacks_wrap, self._TOOLS)
        assert result is inner
        assert inner.bound_tools == self._TOOLS

    def test_real_runnable_retry_wrapper(self) -> None:
        """Integration: safe_bind_tools works through a real RunnableRetry from create_llm."""
        # create_llm without tools= wraps in RunnableRetry (max_retries=2)
        llm = create_llm(_settings(llm_max_retries=2))
        # Confirm the outer wrapper has no bind_tools
        assert not hasattr(type(llm), "bind_tools") or not callable(getattr(type(llm), "bind_tools", None))
        # safe_bind_tools must succeed — binding at the base level
        bound = safe_bind_tools(llm, self._TOOLS)
        kwargs = getattr(bound, "kwargs", {})
        assert "tools" in kwargs, "tools must be in kwargs after safe_bind_tools"
        assert kwargs["tools"] == self._TOOLS

    def test_prebound_llm_rebind_preserves_retry_wrapper(self) -> None:
        """wiki_runner pre-binds with the union of planner + writer schemas.
        Each agent then calls safe_bind_tools with its narrower subset; the
        idempotency short-circuit returns the pre-bound LLM as-is, preserving
        the outer ``RunnableRetry`` wrapper that sits underneath the binding.
        """
        from langchain_core.runnables.retry import RunnableRetry

        extra_tool = {
            "type": "function",
            "function": {"name": "extra", "description": "x", "parameters": {}},
        }
        pre_bound = create_llm(
            _settings(llm_max_retries=2),
            tools=self._TOOLS + [extra_tool],
        )
        assert hasattr(pre_bound, "kwargs")
        assert isinstance(pre_bound.bound, RunnableRetry), (
            "Pre-binding must wrap the retry inside the binding (retry preserved)."
        )

        # Agent rebinds with its narrower schemas (subset of the union).
        rebound = safe_bind_tools(pre_bound, self._TOOLS)
        # Idempotency: the subset is covered, so the pre-bound LLM is returned as-is.
        assert rebound is pre_bound, (
            "safe_bind_tools must be idempotent when desired tools are already bound; "
            "rebinding strips the RunnableRetry wrapper under _ChatModelBinding."
        )
        assert isinstance(rebound.bound, RunnableRetry)

    def test_rebind_when_desired_tools_not_in_existing(self) -> None:
        """If the requested tools are NOT a subset of the existing binding,
        safe_bind_tools rebinds. (Loses retry, but that's the safety trade-off
        when the contract is genuinely different.)"""
        existing_tool = {
            "type": "function",
            "function": {"name": "only_this_tool", "description": "x", "parameters": {}},
        }
        pre_bound = create_llm(_settings(llm_max_retries=2), tools=[existing_tool])
        # Agent wants a different tool not in the existing binding.
        rebound = safe_bind_tools(pre_bound, self._TOOLS)
        assert rebound is not pre_bound
        assert rebound.kwargs["tools"] == self._TOOLS
