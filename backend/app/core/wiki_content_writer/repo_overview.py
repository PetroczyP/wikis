"""Repository overview page generator (#feat).

Generates a high-level overview page for the wiki root (_index.md) by
reading actual wiki page markdown content and synthesising a narrative
overview with the LLM.  This replaces the single-call summarisation over
page titles/descriptions.

Public API
----------
``generate_repo_overview(pages, llm, repo_name, repo_url, branch, repo_analysis)``
    → ``str`` — markdown for the repository overview page.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Maximum number of pages whose markdown is read in Step 1
_MAX_PAGES_TO_READ = 5

# Maximum characters of markdown per page (context budget)
_MAX_CHARS_PER_PAGE = 2000

_OVERVIEW_SYSTEM_PROMPT = """\
You are a technical writer creating a concise, engaging repository overview page for a software wiki.

You will be given:
1. An architectural analysis of the codebase (background context).
2. Excerpts from up to {max_pages} generated wiki pages (the actual content).

Your task: synthesise this into a well-structured overview page with ALL of the following sections:

## Purpose
What does this software do? What problem does it solve? (2-3 sentences)

## How It Works
Step-by-step narrative of the main data flow or request lifecycle. (3-5 sentences)

## Key Capabilities
Bullet list of 5-8 key features or capabilities.

## Architecture
Brief description of the main components and how they relate. (3-5 sentences)

## Getting Started
Pointers to the most important wiki pages for a new engineer.
Use [[Page Title]] wikilinks (e.g. [[Authentication]], [[API Reference]]).

## Wiki Pages
List every page as a wikilink bullet with a one-sentence description.
Format: - [[Page Title]] — description

Rules:
- Do NOT fabricate claims about the codebase.
- Be concise. Aim for 500-800 words total.
- Output raw markdown only — no code fences, no preamble.
"""


def _get_page_attr(page: Any, attr: str, default: str = "") -> str:
    """Extract a string attribute from either a dataclass/object or a dict."""
    if isinstance(page, dict):
        return page.get(attr, default) or default
    return getattr(page, attr, default) or default


def _build_page_excerpts(pages: list[Any], max_pages: int, max_chars: int) -> str:
    """Read markdown excerpts from the first ``max_pages`` pages.

    Prefers the ``markdown`` field (present on ``GeneratedPage``).  Falls
    back to ``description`` when markdown is absent or empty.
    """
    excerpts: list[str] = []
    for page in pages[:max_pages]:
        title = _get_page_attr(page, "title", "Untitled")
        markdown = _get_page_attr(page, "markdown")
        if not markdown:
            markdown = _get_page_attr(page, "description")
        excerpt = (markdown or "")[:max_chars]
        if excerpt:
            excerpts.append(f"### {title}\n{excerpt}")
    return "\n\n".join(excerpts)


def _build_pages_context(pages: list[Any], repo_name: str) -> str:
    """Build a compact context string from page titles and descriptions (fallback)."""
    lines: list[str] = [
        f"Repository: {repo_name}",
        "",
        "Wiki pages generated:",
    ]
    for page in pages:
        title = _get_page_attr(page, "title", "Untitled")
        description = _get_page_attr(page, "description")
        lines.append(f"- {title}: {description}" if description else f"- {title}")
    return "\n".join(lines)


def _build_page_list_md(pages: list[Any]) -> str:
    """Build a '## Wiki Pages' bullet list as a fallback if the LLM call fails."""
    lines: list[str] = ["## Wiki Pages", ""]
    for page in pages:
        title = _get_page_attr(page, "title", "Untitled")
        description = _get_page_attr(page, "description")
        if description:
            lines.append(f"- [[{title}]] — {description}")
        else:
            lines.append(f"- [[{title}]]")
    return "\n".join(lines)


def generate_repo_overview(
    pages: list[Any],
    llm: "BaseChatModel",
    repo_name: str,
    repo_url: str,
    branch: str = "main",
    repo_analysis: str = "",
) -> str:
    """Generate a high-level repository overview page.

    Two-step approach:
    1. Read actual markdown content from up to 5 generated wiki pages.
    2. LLM synthesises a proper narrative overview with structured sections.

    The ``repo_analysis`` document (from ``RepoExplorer``) is included in
    the system prompt context to enable prompt caching.

    Parameters
    ----------
    pages:
        List of ``GeneratedPage`` (or any object/dict with ``title``,
        ``description``, and optionally ``markdown`` attributes) produced by
        the unified pipeline.
    llm:
        LangChain ``BaseChatModel`` used for the overview generation.
        The low-tier LLM (``llm_low``) is recommended — this is a cheap call.
    repo_name:
        Human-readable repository name (e.g. ``"wikis"``).
    repo_url:
        Repository URL for reference (kept for future use / logging).
    branch:
        Branch name (default ``"main"``).  Kept for future use.
    repo_analysis:
        Architectural analysis document from ``RepoExplorer``.  Included in
        the system prompt for richer context (enables prompt caching when the
        same analysis is reused).

    Returns
    -------
    str
        Markdown content for the overview page.  Falls back to a simple
        page list if the LLM call fails.
    """
    if not pages:
        return f"# {repo_name}\n\nNo wiki pages were generated for this repository.\n"

    # ── Step 1: build page context (excerpts from real markdown) ─────────────
    page_excerpts = _build_page_excerpts(pages, _MAX_PAGES_TO_READ, _MAX_CHARS_PER_PAGE)
    if not page_excerpts:
        # Fall back to titles+descriptions if no markdown is available
        page_excerpts = _build_pages_context(pages, repo_name)

    # Build full page list for the "## Wiki Pages" section
    all_pages_list = "\n".join(
        f"- {_get_page_attr(p, 'title', 'Untitled')}: {_get_page_attr(p, 'description')}"
        for p in pages
    )

    system_prompt = _OVERVIEW_SYSTEM_PROMPT.format(max_pages=_MAX_PAGES_TO_READ)

    # Prepend the architectural analysis to the system prompt for caching.
    if repo_analysis:
        system_prompt = (
            f"## Architectural Context\n{repo_analysis[:3000]}\n\n---\n\n"
            + system_prompt
        )

    user_prompt = (
        f"Write a repository overview page for `{repo_name}`.\n\n"
        f"## Page Excerpts\n{page_excerpts}\n\n"
        f"## All Pages (for the Wiki Pages section)\n{all_pages_list}\n\n"
        "Follow the system prompt instructions exactly. Include all sections."
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        overview_body = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        ).strip()

        if not overview_body:
            raise ValueError("LLM returned empty content")

        # Ensure the page starts with the repo title heading.
        if overview_body.startswith(f"# {repo_name}"):
            return overview_body + "\n"
        return f"# {repo_name}\n\n{overview_body}\n"

    except Exception as exc:
        logger.warning(
            "[repo_overview] LLM call failed for repo=%r: %s — falling back to page list",
            repo_name,
            exc,
        )
        page_list = _build_page_list_md(pages)
        return (
            f"# {repo_name}\n\n"
            f"This wiki covers the `{repo_name}` repository.\n\n"
            f"{page_list}\n"
        )
