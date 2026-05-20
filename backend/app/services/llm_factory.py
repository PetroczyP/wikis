"""
Configurable LLM and embedding model factory.

Creates LangChain-compatible LLM and embedding instances from app settings,
supporting multiple providers (OpenAI, Anthropic, custom OpenAI-compatible,
GitHub Copilot, GitHub Models).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from app.config import Settings

logger = logging.getLogger(__name__)


def make_cached_system_message(content: str, cache: bool = True) -> SystemMessage:
    """Build a SystemMessage, optionally with a Bedrock cachePoint checkpoint.

    On Bedrock (ChatBedrockConverse), appending ``{"cachePoint": {"type": "default"}}``
    as the last content block marks everything before it as cacheable across calls
    that share the same system prompt prefix — reducing token cost on repeated
    agentic tool-loop rounds.

    On all other providers the extra block is omitted and a plain string
    ``SystemMessage`` is returned instead (no-op for OpenAI / Anthropic direct
    / Ollama).

    Parameters
    ----------
    content:
        System prompt text.
    cache:
        When ``True`` (the default), wraps content in a list with a trailing
        ``cachePoint`` block.  Pass ``settings.llm_provider == "bedrock"`` at
        each call site, or use :func:`is_bedrock_llm` to detect from the LLM
        instance when settings are not available.
    """
    if not cache:
        return SystemMessage(content=content)
    return SystemMessage(content=[
        {"type": "text", "text": content},
        {"cachePoint": {"type": "default"}},
    ])


def is_bedrock_llm(llm: Any) -> bool:
    """Return ``True`` when *llm* (or its inner LLM) is a ``ChatBedrockConverse``.

    Walks ``RunnableRetry`` (.bound) and ``RunnableWithFallbacks`` (.runnable)
    chains — the same structure that :func:`safe_bind_tools` navigates — so the
    check works whether or not the LLM is wrapped.
    """
    try:
        from langchain_aws import ChatBedrockConverse  # noqa: PLC0415
    except ImportError:
        return False

    cursor: Any = llm
    for _ in range(6):  # bounded walk
        if cursor is None:
            break
        if isinstance(cursor, ChatBedrockConverse):
            return True
        cursor = (
            getattr(cursor, "bound", None)
            or getattr(cursor, "runnable", None)
        )
    return False


def _bedrock_credentials(settings: Settings) -> dict[str, str]:
    """Build explicit boto3 credentials from settings. Raises if missing."""
    creds = {}
    if settings.aws_access_key_id:
        creds["aws_access_key_id"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        creds["aws_secret_access_key"] = settings.aws_secret_access_key.get_secret_value()
    if not creds:
        logger.warning("Bedrock: no explicit AWS creds — falling back to boto3 default chain")
    return creds


def create_llm(
    settings: Settings,
    tier: str = "high",
    tools: list | None = None,
    **overrides: Any,
) -> BaseChatModel:
    """Create an LLM instance from settings.

    Supports providers: openai, anthropic, custom, ollama, gemini, bedrock, copilot, github.
    Tier: "high" (default, page generation) or "low" (quality check, enhancement).
    Overrides: model, temperature, max_tokens, streaming.

    Parameters
    ----------
    tools:
        Optional list of tool schemas (OpenAI function format).  When provided,
        ``bind_tools`` is called on the *base* LLM before any retry/fallback
        wrappers are applied so tool-call routing is preserved through them.
    """
    from langchain_openai import ChatOpenAI

    skip_retry = overrides.pop("skip_retry", False)
    provider = overrides.pop("provider", settings.llm_provider)

    # Resolve model from tier
    if "model" in overrides:
        model = overrides.pop("model")
    elif tier == "low" and settings.llm_model_low:
        model = settings.llm_model_low
    else:
        model = settings.llm_model
    api_key = settings.llm_api_key.get_secret_value()
    base_url = overrides.pop("base_url", settings.llm_api_base)

    # Temperature heuristic: reasoning models (o1, o3, etc.) need temperature=1.0
    # Use basename to handle namespaced models like "openai/gpt-4o"
    model_basename = model.rsplit("/", 1)[-1]
    default_temp = 1.0 if model_basename.startswith("o") else 0.1
    temperature = overrides.pop("temperature", default_temp)
    max_tokens = overrides.pop("max_tokens", 64000)
    streaming = overrides.pop("streaming", False)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **overrides,
        )
    elif provider in ("openai", "custom"):
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **overrides,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=settings.ollama_num_ctx,
            **overrides,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_key = settings.gemini_api_key
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY required for gemini provider")
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=gemini_key.get_secret_value(),
            temperature=temperature,
            max_output_tokens=max_tokens,
            **overrides,
        )
    elif provider == "copilot":
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=settings.github_copilot_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **overrides,
        )
    elif provider == "github":
        # GitHub Models: Claude models use Anthropic API, everything else OpenAI-compatible
        if model.startswith(("claude-", "anthropic/")):
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                model=model,
                api_key=api_key,
                base_url=settings.github_models_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                **overrides,
            )
        else:
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=settings.github_models_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                **overrides,
            )
    elif provider == "bedrock":
        import botocore.config
        from langchain_aws import ChatBedrockConverse

        bedrock_kwargs = _bedrock_credentials(settings)
        bedrock_config = botocore.config.Config(
            read_timeout=settings.llm_timeout, connect_timeout=30, retries={"max_attempts": 3}
        )
        llm = ChatBedrockConverse(
            model_id=model,
            region_name=settings.aws_region,
            temperature=temperature,
            max_tokens=max_tokens,
            config=bedrock_config,
            **bedrock_kwargs,
            **overrides,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    # Resolve effective base_url for logging
    effective_base_url = base_url
    if provider == "copilot":
        effective_base_url = settings.github_copilot_base_url
    elif provider == "github":
        effective_base_url = settings.github_models_base_url
    logger.info(f"Created LLM: provider={provider}, model={model}, base_url={effective_base_url}")

    # Bind tools on the base LLM BEFORE wrapping with retry/fallbacks.
    # RunnableRetry does not proxy bind_tools, so binding after wrapping silently drops tools.
    if tools:
        llm = llm.bind_tools(tools)

    # Add retry with exponential backoff for transient errors
    # Skip for ask/research engines — RunnableRetry breaks _llm_type access
    max_retries = settings.llm_max_retries
    if max_retries > 0 and not skip_retry:
        llm = llm.with_retry(
            retry_if_exception_type=(Exception,),
            wait_exponential_jitter=True,
            stop_after_attempt=max_retries,
        )
        logger.info(f"LLM retry enabled: max_retries={max_retries}")

    # Add fallback chain if configured
    if settings.llm_fallback_models:
        fallback_models = [m.strip() for m in settings.llm_fallback_models.split(",") if m.strip()]
        fallback_provider = settings.llm_fallback_provider or provider
        fallbacks = []
        for fb_model in fallback_models:
            try:
                fb_llm = create_llm(
                    Settings(
                        **{
                            **settings.model_dump(),
                            "llm_provider": fallback_provider,
                            "llm_model": fb_model,
                            "llm_fallback_models": None,  # prevent recursion
                            "llm_max_retries": min(max_retries, 2),  # reduced retries for fallbacks
                        }
                    ),
                )
                fallbacks.append(fb_llm)
                logger.info(f"Fallback model added: {fallback_provider}/{fb_model}")
            except Exception as e:
                logger.warning(f"Failed to create fallback {fb_model}: {e}")
        if fallbacks:
            llm = llm.with_fallbacks(fallbacks)
            logger.info(f"Fallback chain: {len(fallbacks)} models")

    return llm


def safe_bind_tools(llm: Any, tools: list) -> Any:
    """Bind tools through retry/fallback wrappers if needed.

    LangChain's ``with_retry()`` returns a ``RunnableRetry`` and
    ``with_fallbacks()`` returns a ``RunnableWithFallbacks``.  Neither proxies
    ``bind_tools``.  This helper walks the ``.bound`` / ``.runnable`` chain to
    find the inner ``BaseChatModel`` (or any object exposing ``bind_tools``) and
    binds there.  The outer retry/fallback wrappers continue to work because
    LangChain's ``Runnable`` composition delegates ``invoke()`` to the inner.

    If ``llm`` already exposes ``bind_tools`` directly (e.g. it is a plain
    ``BaseChatModel`` or ``_ChatModelBinding``), the call is delegated
    immediately without unwrapping.

    Parameters
    ----------
    llm:
        An LLM or wrapped LLM (``RunnableRetry``, ``RunnableWithFallbacks``,
        or bare ``BaseChatModel``).
    tools:
        Tool schemas or ``@tool``-decorated callables.

    Returns
    -------
    Any
        The result of ``bind_tools(tools)`` called on the innermost LLM that
        supports it.  The returned object is placed where ``llm`` was; outer
        wrappers (retry/fallback) are NOT re-applied because binding replaces
        the inner runnable, not the outer shell.

    Raises
    ------
    TypeError
        When no inner LLM with ``bind_tools`` can be found.
    """
    # Idempotency check: if any layer of the LLM wrapper chain already has a
    # binding whose tools cover the requested schemas, return as-is. Re-calling
    # ``bind_tools()`` on a ``_ChatModelBinding`` walks down to the BASE LLM
    # and binds there, stripping any intermediate ``RunnableRetry`` wrapper.
    # The wiki_runner pre-binds with the canonical WIKI_TOOL_SCHEMAS catalog
    # so each agent's rebind can be skipped here — preserving retry semantics.
    #
    # Walk the chain because the binding may sit BELOW the retry wrapper when
    # construction order is ``bind_tools → with_retry`` (the create_llm path
    # with tools= passed). In that case the top-level object is the retry
    # wrapper with no ``kwargs.tools`` of its own.
    desired_names = _tool_names(tools)
    if desired_names:
        cursor = llm
        for _ in range(5):  # bounded walk — wrapper chains are shallow
            if cursor is None:
                break
            existing = (
                getattr(cursor, "kwargs", {}).get("tools")
                if hasattr(cursor, "kwargs") else None
            )
            if existing:
                if desired_names.issubset(_tool_names(existing)):
                    return llm
                break  # binding found but tools differ — fall through to rebind
            cursor = (
                getattr(cursor, "bound", None)
                or getattr(cursor, "runnable", None)
            )

    if hasattr(llm, "bind_tools"):
        return llm.bind_tools(tools)

    # Walk retry (.bound) and fallback (.runnable) chains.
    inner = llm
    while inner is not None:
        bound = getattr(inner, "bound", None)
        candidate = bound if bound is not None else getattr(inner, "runnable", None)
        if candidate is None:
            break
        if hasattr(candidate, "bind_tools"):
            return candidate.bind_tools(tools)
        inner = candidate

    raise TypeError(
        f"Cannot bind_tools through {type(llm).__name__}: "
        "no inner LLM with bind_tools found in the .bound/.runnable chain"
    )


def _tool_names(tools: list) -> set[str]:
    """Extract tool names from a heterogeneous list of tool descriptors.

    Handles three observed shapes: OpenAI ``{type, function: {name, ...}}``,
    Bedrock flattened ``{name, description, input_schema}`` (what bind_tools
    converts OpenAI schemas into on Bedrock), and ``@tool``-decorated callables.
    """
    names: set[str] = set()
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function")
            if isinstance(fn, dict) and fn.get("name"):
                names.add(fn["name"])
            elif t.get("name"):
                names.add(t["name"])
        else:
            name = getattr(t, "name", None)
            if name:
                names.add(name)
    return names


def create_embeddings(settings: Settings, **overrides: Any) -> Embeddings:
    """Create an embeddings instance from settings.

    Supports providers: openai, custom, ollama, gemini, bedrock, github.
    Raises ValueError for unsupported providers or missing credentials.
    """
    model = overrides.pop("model", settings.embedding_model)
    provider = overrides.pop("provider", settings.embedding_provider or settings.llm_provider)

    # Anthropic and Copilot have no embedding API — auto-fallback to openai if embedding_api_key is set
    if provider in ("anthropic", "copilot"):
        if settings.embedding_api_key:
            logger.info(f"Provider {provider} has no embedding API — falling back to openai for embeddings")
            provider = "openai"
        else:
            raise ValueError(
                f"{provider.title()} has no embedding API — set EMBEDDING_PROVIDER and EMBEDDING_API_KEY (e.g. openai)"
            )

    # Resolve embedding API key (falls back to LLM key)
    embedding_key = settings.embedding_api_key
    if embedding_key:
        api_key = embedding_key.get_secret_value()
    else:
        api_key = settings.llm_api_key.get_secret_value()

    base_url = overrides.pop("base_url", settings.embedding_api_base)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model=model or "nomic-embed-text", base_url=settings.ollama_base_url, **overrides)
        logger.info(f"Created embeddings: Ollama model={model}")
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        gemini_key = settings.gemini_api_key
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY required for gemini embeddings")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model or "models/embedding-001", google_api_key=gemini_key.get_secret_value(), **overrides
        )
        logger.info(f"Created embeddings: Gemini model={model}")
    elif provider == "bedrock":
        from langchain_aws import BedrockEmbeddings

        bedrock_kwargs = _bedrock_credentials(settings)
        embeddings = BedrockEmbeddings(
            model_id=model or "amazon.titan-embed-text-v1",
            region_name=settings.aws_region,
            **bedrock_kwargs,
            **overrides,
        )
        logger.info(f"Created embeddings: Bedrock model={model}")
    elif provider == "github":
        from langchain_openai.embeddings import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=model or "text-embedding-3-small",
            api_key=api_key,
            base_url=settings.github_models_base_url,
            **overrides,
        )
        logger.info(f"Created embeddings: GitHub Models model={model}")
    elif api_key:
        from langchain_openai.embeddings import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url, **overrides)
        logger.info(f"Created embeddings: OpenAI model={model}, base_url={base_url}")
    else:
        raise ValueError("No API key available for embeddings — configure llm_api_key or embedding_api_key")
    return embeddings
