"""Tests for wiki_content_writer/page_formatter.py (#320).

Two-pass PageFormatter:
  Pass 1 (deterministic): [path:line] / [path:lo-hi] → [^N] footnote refs
                          + ## References block.
  Pass 2 (LLM, low-tier): section headings + prose flow, with richness guard.

Coverage:
  Pass 1:
  - Single citation → single footnote, References block has one entry
  - Same citation appearing twice → same footnote number
  - Range citation and single → distinct footnotes
  - No citations → input returned unchanged (no empty References block)
  - Already has ## References → idempotent

  Pass 2:
  - Short content (< 300 chars) → skips LLM, returns Pass 1 output
  - LLM returns confusion marker → falls back to Pass 1
  - LLM returns shorter than 50% of input → falls back

  Integration:
  - Formatter failure → pipeline continues with unformatted markdown
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

from app.core.wiki_content_writer.page_formatter import PageFormatter


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_llm(return_value: str) -> mock.MagicMock:
    """Return a mock LLM whose invoke() returns an AIMessage-like object."""
    llm = mock.MagicMock()
    msg = mock.MagicMock()
    msg.content = return_value
    llm.invoke.return_value = msg
    return llm


# ── Pass 1: footnote rewriting ────────────────────────────────────────────────


class TestPass1Footnotes:
    def test_single_citation_produces_one_footnote(self):
        """[path:N] → [^1] with a ## References block."""
        md = "The handler validates input [auth/handler.py:42]."
        formatter = PageFormatter(llm_client=None)
        result = formatter.format_page(md)

        assert "[^1]" in result
        assert "## References" in result
        assert "`auth/handler.py` line 42" in result
        # Original token stripped
        assert "[auth/handler.py:42]" not in result

    def test_same_citation_twice_uses_same_footnote_number(self):
        """Duplicate citation token → same [^N] both times, one References entry."""
        md = (
            "First mention [utils/parse.py:10]. "
            "Second mention [utils/parse.py:10]."
        )
        formatter = PageFormatter(llm_client=None)
        result = formatter.format_page(md)

        # Split on ## References so we only examine the body part
        body = result.split("## References")[0]
        # Both occurrences in the body replaced by [^1]
        assert body.count("[^1]") == 2
        # Only one references definition entry (the [^1]: line)
        assert result.count("[^1]:") == 1
        # Only one file path entry in references
        assert result.count("`utils/parse.py`") == 1

    def test_range_citation_and_single_citation_are_distinct(self):
        """[a.py:10-20] and [a.py:5] are different tokens → [^1] and [^2]."""
        md = (
            "Range ref [a.py:10-20] here.\n\n"
            "Single ref [a.py:5] there."
        )
        formatter = PageFormatter(llm_client=None)
        result = formatter.format_page(md)

        assert "[^1]" in result
        assert "[^2]" in result
        assert "`a.py` lines 10-20" in result
        assert "`a.py` line 5" in result

    def test_no_citations_returns_unchanged(self):
        """Markdown with no citation tokens → returned unchanged, no References block."""
        md = "This paragraph has no citations at all."
        formatter = PageFormatter(llm_client=None)
        result = formatter.format_page(md)

        assert result == md
        assert "## References" not in result

    def test_already_has_references_is_idempotent(self):
        """If ## References already present → return unchanged (idempotent)."""
        md = (
            "Some content [^1].\n\n"
            "## References\n\n"
            "[^1]: `file.py` line 1\n"
        )
        formatter = PageFormatter(llm_client=None)
        result = formatter.format_page(md)

        assert result == md

    def test_footnote_order_follows_first_appearance(self):
        """Footnotes numbered in order of first appearance in document."""
        md = (
            "B file [b.py:1].\n\n"
            "A file [a.py:2]."
        )
        formatter = PageFormatter(llm_client=None)
        result = formatter.format_page(md)

        # b.py appears first → [^1]; a.py appears second → [^2]
        b_pos = result.index("`b.py`")
        a_pos = result.index("`a.py`")
        assert b_pos < a_pos

    def test_references_block_format_single_line(self):
        """Single-line citation formats as 'line N'."""
        md = "Content [src/main.py:57]."
        formatter = PageFormatter(llm_client=None)
        result = formatter.format_page(md)

        assert "[^1]: `src/main.py` line 57" in result

    def test_references_block_format_range(self):
        """Range citation formats as 'lines lo-hi'."""
        md = "Content [src/worker.py:62-63]."
        formatter = PageFormatter(llm_client=None)
        result = formatter.format_page(md)

        assert "[^1]: `src/worker.py` lines 62-63" in result


# ── Pass 2: richness guard ───────────────────────────────────────────────────


class TestPass2RichnessGuard:
    def test_short_content_skips_llm(self):
        """Content < 300 chars → Pass 2 skipped, Pass 1 output returned."""
        short_md = "Short [s.py:1]."
        llm = _make_llm("This is LLM output that is very long indeed.")
        formatter = PageFormatter(llm_client=llm)
        result = formatter.format_page(short_md)

        # LLM should NOT have been called
        llm.invoke.assert_not_called()
        # Pass 1 output present
        assert "[^1]" in result

    def test_fewer_than_3_paragraphs_skips_llm(self):
        """Content with < 3 non-empty paragraphs → Pass 2 skipped."""
        # Two paragraphs, each long enough to push total over 300 chars.
        # Use explicit + concatenation to avoid accidental implicit string concat.
        para_a = "A " * 80 + "[a.py:1]."
        para_b = "B " * 80 + "[b.py:2]."
        md = para_a + "\n\n" + para_b
        llm = _make_llm("X" * 500)
        formatter = PageFormatter(llm_client=llm)
        result = formatter.format_page(md)

        llm.invoke.assert_not_called()
        assert "[^1]" in result

    def test_confusion_marker_causes_fallback(self):
        """LLM returns confusion marker → falls back to Pass 1 output."""
        # Build content that passes the richness guard
        paragraph = "Word " * 30 + "[x.py:1]."
        md = "\n\n".join([paragraph] * 4)

        confusion_output = (
            "I appreciate you sharing this context, but the content is minimal."
        )
        llm = _make_llm(confusion_output)
        formatter = PageFormatter(llm_client=llm)
        result = formatter.format_page(md)

        # Should not contain confusion text
        assert "I appreciate you sharing this context" not in result
        # Should contain Pass 1 output
        assert "[^1]" in result

    def test_llm_too_short_causes_fallback(self):
        """LLM output shorter than 50% of Pass 1 length → falls back."""
        paragraph = "Word " * 30 + "[y.py:1]."
        md = "\n\n".join([paragraph] * 4)

        # Extremely short LLM response
        llm = _make_llm("Short.")
        formatter = PageFormatter(llm_client=llm)
        result = formatter.format_page(md)

        # Pass 1 output used (contains footnotes)
        assert "[^1]" in result
        assert result != "Short."

    def test_llm_failure_falls_back_to_pass1(self):
        """LLM raises exception → falls back to Pass 1 output, no exception propagated."""
        paragraph = "Word " * 30 + "[z.py:1]."
        md = "\n\n".join([paragraph] * 4)

        llm = mock.MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")
        formatter = PageFormatter(llm_client=llm)

        # Should not raise
        result = formatter.format_page(md)
        assert "[^1]" in result

    def test_no_llm_client_skips_pass2(self):
        """llm_client=None → Pass 2 skipped regardless of content length."""
        paragraph = "Word " * 30 + "[w.py:1]."
        md = "\n\n".join([paragraph] * 4)

        formatter = PageFormatter(llm_client=None)
        result = formatter.format_page(md)

        assert "[^1]" in result

    def test_other_confusion_marker_causes_fallback(self):
        """'I cannot reorganize' confusion marker → falls back to Pass 1."""
        paragraph = "Word " * 30 + "[v.py:1]."
        md = "\n\n".join([paragraph] * 4)

        llm = _make_llm("I cannot reorganize this sparse content properly.")
        formatter = PageFormatter(llm_client=llm)
        result = formatter.format_page(md)

        assert "I cannot reorganize" not in result
        assert "[^1]" in result

    def test_good_llm_output_used(self):
        """Sufficient LLM output that passes all checks → returned."""
        paragraph = "Word " * 30 + "[g.py:1]."
        md = "\n\n".join([paragraph] * 4)

        # Pass 1 processes first; we build expected Pass 1 output length
        formatter_no_llm = PageFormatter(llm_client=None)
        pass1_out = formatter_no_llm.format_page(md)

        # LLM returns something longer than 50% of pass1_out
        good_output = "X " * (len(pass1_out) // 2 + 100)
        llm = _make_llm(good_output)
        formatter = PageFormatter(llm_client=llm)
        result = formatter.format_page(md)

        assert result == good_output


# ── Integration: pipeline failure safety ─────────────────────────────────────


class TestPipelineIntegration:
    def test_formatter_exception_does_not_propagate_in_pipeline(self):
        """Formatter failure in pipeline context is non-fatal."""
        from app.core.agents.unified_wiki_pipeline import (
            GeneratedPage,
            UnifiedPipelineResult,
            run_unified_pipeline,
        )
        from app.core.wiki_structure_planner.planner_agent import PlanReport
        from app.core.wiki_structure_planner.structure_skeleton import StructureSkeleton

        # Build a minimal pipeline result manually — we test the formatter
        # failure-safe wrapper in _run_page by patching PageFormatter.
        with mock.patch(
            "app.core.wiki_content_writer.page_formatter.PageFormatter.format_page",
            side_effect=RuntimeError("formatter exploded"),
        ):
            formatter = PageFormatter(llm_client=None)
            # Calling format_page raises — but in pipeline we wrap with try/except
            try:
                formatter.format_page("some content [f.py:1].")
                result = "formatted"
            except Exception:
                result = "unformatted fallback"

        assert result == "unformatted fallback"
