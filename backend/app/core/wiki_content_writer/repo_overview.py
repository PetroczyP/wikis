"""Repository overview page generator (#feat).

Generates a high-level overview page for the wiki root (_index.md) by
summarising the generated wiki pages with an LLM call.  This replaces the
static "Auto-generated from N cluster(s)" placeholder that ``_generate_wiki_structure``
previously produced for the wiki root.

Public API
----------
``generate_repo_overview(pages, llm, repo_name, repo_url, branch)``
    → ``str`` — markdown for the repository overview page.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_OVERVIEW_SYSTEM_PROMPT = """\
You are a technical writer creating a concise, engaging repository overview page for a software wiki.
You will be given a list of wiki pages with their titles and one-sentence descriptions.
Your task is to synthesise this into a well-structured overview page.

Rules:
- Write 3-5 paragraphs covering: what the software does, problems it solves, who uses it, and its main architectural areas.
- After the overview prose, add a "## Main Areas" section that groups related pages and explains how they fit together.
- After "## Main Areas", add a "## Wiki Pages" section listing every page as a wikilink bullet with a one-sentence description.
- Use [[Page Title]] syntax for wikilinks.
- Do NOT fabricate claims about the codebase beyond what the page summaries imply.
- Be concise. Aim for 400-700 words total (prose + lists).
- Output raw markdown only — no code fences, no preamble.
"""


def _get_page_attr(page: Any, attr: str, default: str = "") -> str:
    """Extract a string attribute from either a dataclass/object or a dict."""
    if isinstance(page, dict):
        return page.get(attr, default) or default
    return getattr(page, attr, default) or default


def _build_pages_context(pages: list[Any], repo_name: str) -> str:
    """Build a compact context string from page titles and descriptions.

    Only titles and descriptions are included — full markdown is too long
    and would blow the context window for a summarisation call.
    """
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
) -> str:
    """Generate a high-level repository overview page using the wiki pages as context.

    Produces a markdown page that explains:
    - What this repository is and what problem it solves
    - The main functional areas (derived from page titles/descriptions)
    - How the areas connect
    - A navigable list of all wiki pages

    Parameters
    ----------
    pages:
        List of ``GeneratedPage`` (or any object/dict with ``title`` and
        ``description`` attributes) produced by the unified pipeline.
    llm:
        LangChain ``BaseChatModel`` used for the overview generation.
        The low-tier LLM (``llm_low``) is recommended — this is a cheap call.
    repo_name:
        Human-readable repository name (e.g. ``"wikis"``).
    repo_url:
        Repository URL for reference (not injected into the LLM prompt to
        avoid hallucination; kept for future use / logging).
    branch:
        Branch name (default ``"main"``).  Kept for future use.

    Returns
    -------
    str
        Markdown content for the overview page.  Falls back to a simple
        page list if the LLM call fails.
    """
    if not pages:
        return f"# {repo_name}\n\nNo wiki pages were generated for this repository.\n"

    context = _build_pages_context(pages, repo_name)
    user_prompt = (
        f"Based on the wiki pages listed below, write a repository overview page "
        f"for the `{repo_name}` repository. Follow the system prompt instructions exactly.\n\n"
        f"{context}"
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        messages = [
            SystemMessage(content=_OVERVIEW_SYSTEM_PROMPT),
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
        # The LLM may or may not include it — normalise either way.
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
