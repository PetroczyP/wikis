"""Tests for chaptered-page frontmatter extraction in routes (#287).

Covers ``_extract_frontmatter_title_section`` reading ``chapter:`` and
``page_order:`` fields that chaptered sub-pages carry.
"""

from __future__ import annotations

import pytest

from app.api.routes import _extract_frontmatter_title_section


class TestFrontmatterExtractionFlat:
    """Backward-compat: existing ``section:`` field still works."""

    def test_section_field_returned(self):
        content = "---\ntitle: My Page\nsection: Architecture\n---\nBody text."
        title, section, page_order = _extract_frontmatter_title_section(content)
        assert title == "My Page"
        assert section == "Architecture"
        assert page_order == 0

    def test_no_frontmatter_returns_empty(self):
        title, section, page_order = _extract_frontmatter_title_section("No frontmatter here.")
        assert title == ""
        assert section == ""
        assert page_order == 0

    def test_unclosed_frontmatter_returns_empty(self):
        content = "---\ntitle: Missing close marker\n"
        title, section, page_order = _extract_frontmatter_title_section(content)
        assert title == ""
        assert section == ""
        assert page_order == 0

    def test_title_only_section_empty(self):
        content = "---\ntitle: Just A Title\n---\nBody."
        title, section, page_order = _extract_frontmatter_title_section(content)
        assert title == "Just A Title"
        assert section == ""
        assert page_order == 0


class TestFrontmatterExtractionChaptered:
    """Chaptered sub-pages carry ``chapter:`` and ``page_order:`` (#287)."""

    def test_chapter_field_used_as_section_when_section_absent(self):
        content = (
            "---\n"
            "title: Test Launch REST API\n"
            "chapter: workers-chapter\n"
            "page_order: 2\n"
            "---\nContent."
        )
        title, section, page_order = _extract_frontmatter_title_section(content)
        assert title == "Test Launch REST API"
        # chapter slug is humanised: workers-chapter → "Workers Chapter"
        assert section == "Workers Chapter"
        assert page_order == 2

    def test_section_takes_precedence_over_chapter(self):
        """``section:`` wins when both are present (backward compat)."""
        content = (
            "---\n"
            "title: Sub Page\n"
            "section: Explicit Section\n"
            "chapter: some-chapter\n"
            "---\nContent."
        )
        title, section, _ = _extract_frontmatter_title_section(content)
        assert section == "Explicit Section"

    def test_page_order_parsed_correctly(self):
        content = (
            "---\n"
            "title: Worker Lifecycle\n"
            "chapter: worker-chapter\n"
            "page_order: 5\n"
            "---\nContent."
        )
        _, _, page_order = _extract_frontmatter_title_section(content)
        assert page_order == 5

    def test_page_order_zero_default_when_absent(self):
        content = (
            "---\n"
            "title: No Order\n"
            "chapter: some-chapter\n"
            "---\nContent."
        )
        _, _, page_order = _extract_frontmatter_title_section(content)
        assert page_order == 0

    def test_chapter_slug_humanised_to_display_name(self):
        """Hyphens in chapter slug become spaces; words are title-cased."""
        content = (
            "---\n"
            "title: A Sub-Page\n"
            "chapter: authentication-service\n"
            "---\nContent."
        )
        _, section, _ = _extract_frontmatter_title_section(content)
        assert section == "Authentication Service"

    def test_chapter_index_page_no_chapter_field(self):
        """Chapter index pages have ``is_chapter_index: true`` but no ``chapter:``."""
        content = (
            "---\n"
            "title: Workers Chapter\n"
            "is_chapter_index: true\n"
            "---\nContent."
        )
        title, section, page_order = _extract_frontmatter_title_section(content)
        assert title == "Workers Chapter"
        # No chapter/section field → empty section (path will supply the display name)
        assert section == ""
        assert page_order == 0

    def test_underscore_chapter_slug_humanised(self):
        """Underscores in chapter slug are also replaced with spaces."""
        content = (
            "---\n"
            "title: My Page\n"
            "chapter: data_pipeline\n"
            "---\nContent."
        )
        _, section, _ = _extract_frontmatter_title_section(content)
        assert section == "Data Pipeline"
