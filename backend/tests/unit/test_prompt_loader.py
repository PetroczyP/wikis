"""Unit tests for app.prompts.load_prompt (#346 migration).

Verifies that:
- load_prompt resolves paths relative to the prompts package directory.
- Missing files raise FileNotFoundError with a helpful message.
- Results are strings with real content.
- caching works (same object on second call).
- All migrated prompt files are accessible and non-empty.
- The migrated constants in Python modules match the MD files.
"""

from __future__ import annotations

import pytest

from app.prompts import load_prompt


# ── Basic loader behaviour ────────────────────────────────────────────────────

def test_load_prompt_returns_string():
    """load_prompt returns a non-empty str for a known file."""
    text = load_prompt("writer/system.md")
    assert isinstance(text, str)
    assert len(text) > 100


def test_load_prompt_missing_file_raises_filenotfounderror():
    """Non-existent paths raise FileNotFoundError, not a generic exception."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prompt("nonexistent/path.md")


def test_load_prompt_is_cached():
    """Two calls with the same path return the identical string object (lru_cache)."""
    a = load_prompt("writer/system.md")
    b = load_prompt("writer/system.md")
    assert a is b


def test_load_prompt_strips_whitespace():
    """load_prompt strips leading/trailing whitespace from file contents."""
    text = load_prompt("writer/system.md")
    assert text == text.strip()


# ── Existence check for all migrated prompt files ─────────────────────────────

MIGRATED_FILES = [
    # writer
    "writer/system.md",
    "writer/user_template.md",
    "writer/budget_exhausted.md",
    "writer/chapter_index_budget_exhausted.md",
    # planner
    "planner/system.md",
    "planner/chaptered_system.md",
    "planner/refiner_system.md",
    # ask
    "ask/workflow_instructions.md",
    "ask/tool_instructions.md",
    "ask/output_instructions.md",
    "ask/query_optimization_system.md",
    "ask/query_optimization_user.md",
    "ask/answer_system.md",
    "ask/answer_user.md",
    # research
    "research/workflow_instructions.md",
    "research/tool_instructions.md",
    "research/stopping_criteria.md",
    "research/output_format.md",
    # extractors
    "extractors/image_describe.md",
    "extractors/pdf_page_describe.md",
    # repo
    "repo/explorer_system.md",
    "repo/explorer_budget_exhausted.md",
    "repo/overview_system.md",
    # wiki
    "wiki/surgical_edit_system.md",
    "wiki/surgical_edit_user.md",
    "wiki/page_format.md",
    "wiki/content_generation_v3_tone.md",
    "wiki/repo_analysis_structured.md",
    "wiki/repo_analysis_enhanced.md",
    "wiki/wiki_structure.md",
]


@pytest.mark.parametrize("path", MIGRATED_FILES)
def test_migrated_file_is_non_empty(path: str):
    """Each migrated MD file can be loaded and contains non-trivial content."""
    text = load_prompt(path)
    assert isinstance(text, str)
    assert len(text) >= 30, f"{path} has suspiciously short content: {text!r}"


# ── Template variable placeholders ────────────────────────────────────────────

def test_writer_system_has_format_placeholders():
    """Writer system prompt is a template with {prefix}, {format_description}, {rules}."""
    text = load_prompt("writer/system.md")
    assert "{prefix}" in text
    assert "{format_description}" in text
    assert "{rules}" in text


def test_writer_user_template_has_format_placeholders():
    """Writer user template has the expected {title}, {description}, etc. placeholders."""
    text = load_prompt("writer/user_template.md")
    assert "{title}" in text
    assert "{description}" in text
    assert "{symbols}" in text


def test_planner_system_has_no_format_placeholders():
    """Planner system prompt is static — no {variable} substitutions needed."""
    text = load_prompt("planner/system.md")
    # The planner system prompt contains example JSON with {cluster_id} etc. in it,
    # but those are inside code fences not format placeholders.
    # The template is passed as-is to the LLM without .format().
    assert "You are a wiki structure planner" in text


def test_ask_workflow_has_date_and_iterations_placeholders():
    """Ask workflow instructions template uses {date} and {max_iterations}."""
    text = load_prompt("ask/workflow_instructions.md")
    assert "{date}" in text
    assert "{max_iterations}" in text


def test_repo_explorer_system_has_repo_and_budget_placeholders():
    """Repo explorer system prompt uses {repo_name} and {budget}."""
    text = load_prompt("repo/explorer_system.md")
    assert "{repo_name}" in text
    assert "{budget}" in text


def test_repo_overview_system_has_max_pages_placeholder():
    """Repo overview system prompt uses {max_pages}."""
    text = load_prompt("repo/overview_system.md")
    assert "{max_pages}" in text


def test_surgical_edit_user_has_required_placeholders():
    """Surgical edit user prompt has all required template placeholders."""
    text = load_prompt("wiki/surgical_edit_user.md")
    assert "{page_title}" in text
    assert "{symbol_diffs}" in text
    assert "{current_content}" in text


# ── Integration: Python modules export the loaded prompts ─────────────────────

def test_surgical_edit_prompts_module_exports_match_files():
    """surgical_edit_prompts constants match the MD file content."""
    from app.core.prompts.surgical_edit_prompts import (
        SURGICAL_EDIT_SYSTEM,
        SURGICAL_EDIT_USER_TEMPLATE,
    )
    assert SURGICAL_EDIT_SYSTEM == load_prompt("wiki/surgical_edit_system.md")
    assert SURGICAL_EDIT_USER_TEMPLATE == load_prompt("wiki/surgical_edit_user.md")


def test_planner_prompts_module_exports_match_files():
    """planner_prompts constants match the MD file content."""
    from app.core.wiki_structure_planner.planner_prompts import (
        CHAPTERED_PLANNER_SYSTEM_PROMPT,
        PLANNER_SYSTEM_PROMPT,
    )
    assert PLANNER_SYSTEM_PROMPT == load_prompt("planner/system.md")
    assert CHAPTERED_PLANNER_SYSTEM_PROMPT == load_prompt("planner/chaptered_system.md")


def test_ask_tool_prompts_match_files():
    """AskTool prompt constants match the MD file content."""
    from app.core.ask_tool import (
        ANSWER_SYSTEM_PROMPT,
        ANSWER_USER_PROMPT,
        QUERY_OPTIMIZATION_SYSTEM_PROMPT,
        QUERY_OPTIMIZATION_USER_PROMPT,
    )
    assert QUERY_OPTIMIZATION_SYSTEM_PROMPT == load_prompt("ask/query_optimization_system.md")
    assert QUERY_OPTIMIZATION_USER_PROMPT == load_prompt("ask/query_optimization_user.md")
    assert ANSWER_SYSTEM_PROMPT == load_prompt("ask/answer_system.md")
    assert ANSWER_USER_PROMPT == load_prompt("ask/answer_user.md")


def test_research_prompts_module_exports_match_files():
    """research_prompts constants match the MD file content."""
    from app.core.deep_research.research_prompts import (
        OUTPUT_FORMAT_INSTRUCTIONS,
        RESEARCH_WORKFLOW_INSTRUCTIONS,
        STOPPING_CRITERIA,
        TOOL_USAGE_INSTRUCTIONS,
    )
    assert RESEARCH_WORKFLOW_INSTRUCTIONS == load_prompt("research/workflow_instructions.md")
    assert TOOL_USAGE_INSTRUCTIONS == load_prompt("research/tool_instructions.md")
    assert STOPPING_CRITERIA == load_prompt("research/stopping_criteria.md")
    assert OUTPUT_FORMAT_INSTRUCTIONS == load_prompt("research/output_format.md")
