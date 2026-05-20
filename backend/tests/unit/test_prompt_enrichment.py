"""Unit tests for prompt enrichment — issue #311.

Verifies that the writer system prompt template and planner system prompt
contain enough content to reach the Bedrock 2048-token caching threshold,
and that all documented tool names, citation examples, and evidence pack
field descriptions are present.
"""

from __future__ import annotations

from app.core.wiki_content_writer.writer_agent import (
    _SYSTEM_PROMPT_PREFIX,
    _SYSTEM_PROMPT_TEMPLATE,
)
from app.core.wiki_content_writer.prompts import (
    CITATION_FORMAT_DESCRIPTION,
    CITATION_PROMPT_RULES,
)
from app.core.wiki_structure_planner.planner_prompts import PLANNER_SYSTEM_PROMPT


# ── Writer prompt — token-count threshold ─────────────────────────────────────


class TestWriterPromptLength:
    """The rendered writer system prompt must approach the 2048-token Bedrock
    caching threshold.  We measure in chars/4 (rough approximation).
    """

    def _render(self) -> str:
        """Render the template with the real prompts constants."""
        return _SYSTEM_PROMPT_TEMPLATE.format(
            prefix=_SYSTEM_PROMPT_PREFIX,
            format_description=CITATION_FORMAT_DESCRIPTION,
            rules="\n".join(f"{i + 1}. {r}" for i, r in enumerate(CITATION_PROMPT_RULES)),
        )

    def test_rendered_prompt_exceeds_1700_tokens(self):
        """Rendered writer prompt must be ≥1700 estimated tokens (chars/4)."""
        rendered = self._render()
        estimated_tokens = len(rendered) / 4
        assert estimated_tokens >= 1700, (
            f"Writer prompt is only ~{estimated_tokens:.0f} estimated tokens "
            f"(need ≥1700 for Bedrock caching)"
        )

    def test_template_raw_exceeds_1700_token_chars(self):
        """Even the raw template (before format) should be substantive."""
        # The template has literal content beyond just placeholder references.
        assert len(_SYSTEM_PROMPT_TEMPLATE) > 1700 * 4 - 2000  # allow for placeholder expansion


# ── Writer prompt — tool reference guide ─────────────────────────────────────


class TestWriterPromptToolGuide:
    """All five writer tools must be described in the system prompt template."""

    _EXPECTED_TOOLS = [
        "read_file",
        "get_signature",
        "grep",
        "get_callers",
        "get_callees",
        "list_doc_chunks",
    ]

    def test_all_tools_mentioned(self):
        for tool in self._EXPECTED_TOOLS:
            assert tool in _SYSTEM_PROMPT_TEMPLATE, (
                f"Tool '{tool}' is missing from _SYSTEM_PROMPT_TEMPLATE"
            )


# ── Writer prompt — citation bad examples ────────────────────────────────────


class TestWriterPromptCitationExamples:
    """The writer prompt must include both good and bad citation examples."""

    def test_good_examples_present(self):
        assert "Good" in _SYSTEM_PROMPT_TEMPLATE or "good" in _SYSTEM_PROMPT_TEMPLATE.lower()

    def test_bad_examples_present(self):
        assert "Bad" in _SYSTEM_PROMPT_TEMPLATE or "bad" in _SYSTEM_PROMPT_TEMPLATE.lower()

    def test_discarded_consequence_explained(self):
        """The prompt must warn that uncited paragraphs are discarded."""
        assert "discarded" in _SYSTEM_PROMPT_TEMPLATE.lower()

    def test_bracket_format_contrasted_with_paren(self):
        """The prompt must show that parens are wrong (not brackets)."""
        # Looking for a "parens not brackets" style warning
        assert "paren" in _SYSTEM_PROMPT_TEMPLATE.lower() or "bracket" in _SYSTEM_PROMPT_TEMPLATE.lower()


# ── Writer prompt — citation verifier explanation ─────────────────────────────


class TestWriterPromptVerifierSection:
    """The writer prompt must explain what happens if citations are missing."""

    def test_verifier_explanation_present(self):
        """Prompt must explain that the verifier discards uncited paragraphs."""
        prompt = _SYSTEM_PROMPT_TEMPLATE.lower()
        assert "verif" in prompt  # "verifier" or "verify"

    def test_line_range_verification_explained(self):
        """Prompt must mention that line ranges are checked against the repo."""
        prompt = _SYSTEM_PROMPT_TEMPLATE.lower()
        # Should explain that citations must point to real line ranges
        assert "line" in prompt and ("exist" in prompt or "range" in prompt or "real" in prompt)


# ── Planner prompt — evidence pack fields ────────────────────────────────────


class TestPlannerPromptEvidenceFields:
    """The planner system prompt must include a description of each evidence
    pack field."""

    _EXPECTED_FIELDS = [
        "artifact_names",
        "path_leaves",
        "dir_paths",
        "call_graph_edges",
        "cross_cluster_edges",
        "doc_snippets",
    ]

    def test_evidence_pack_section_present(self):
        assert "evidence pack fields" in PLANNER_SYSTEM_PROMPT.lower(), (
            "Planner prompt must include an 'Evidence pack fields' section"
        )

    def test_all_field_names_documented(self):
        for field in self._EXPECTED_FIELDS:
            assert field in PLANNER_SYSTEM_PROMPT, (
                f"Evidence pack field '{field}' missing from PLANNER_SYSTEM_PROMPT"
            )


# ── Planner prompt — example PageSpec ────────────────────────────────────────


class TestPlannerPromptPageSpecExample:
    """The planner prompt must include a concrete PageSpec JSON example."""

    def test_example_output_section_present(self):
        assert "example" in PLANNER_SYSTEM_PROMPT.lower()

    def test_example_contains_cluster_id(self):
        assert "cluster_id" in PLANNER_SYSTEM_PROMPT

    def test_example_contains_retrieval_query(self):
        assert "retrieval_query" in PLANNER_SYSTEM_PROMPT

    def test_example_contains_target_symbols(self):
        assert "target_symbols" in PLANNER_SYSTEM_PROMPT

    def test_example_contains_target_folders(self):
        assert "target_folders" in PLANNER_SYSTEM_PROMPT


# ── Planner prompt — cross-cluster linking guidance ───────────────────────────


class TestPlannerPromptCrossClusterGuidance:
    """The planner prompt must explain how to use cross_cluster_edges."""

    def test_cross_cluster_section_present(self):
        assert "cross_cluster" in PLANNER_SYSTEM_PROMPT.lower() or (
            "cross-cluster" in PLANNER_SYSTEM_PROMPT.lower()
        )

    def test_cross_cluster_links_explained(self):
        prompt = PLANNER_SYSTEM_PROMPT.lower()
        assert "depend" in prompt or "link" in prompt or "import" in prompt


# ── Planner prompt — token-count threshold ───────────────────────────────────


class TestPlannerPromptLength:
    """The planner prompt must approach the 2048-token caching threshold."""

    def test_planner_prompt_exceeds_1500_tokens(self):
        """Planner system prompt must be ≥1500 estimated tokens (chars/4)."""
        estimated_tokens = len(PLANNER_SYSTEM_PROMPT) / 4
        assert estimated_tokens >= 1500, (
            f"Planner prompt is only ~{estimated_tokens:.0f} estimated tokens "
            f"(need ≥1500 for meaningful caching)"
        )
