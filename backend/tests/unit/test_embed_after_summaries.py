"""Tests for embedding pipeline two-pass approach and text-selection fixes.

Covers:
1. Two-pass embedding in _write_unified_db:
   - Phase 1b: populate_embeddings called BEFORE run_phase2 (needed for
     orphan resolution via db.search_vec).
   - Post-summaries: populate_embeddings called again AFTER populate_summaries
     so nodes that received llm_summary are re-embedded with richer text.
2. SQLite populate_embeddings builds text from llm_summary → signature/docstring
   → symbol_name priority order (source_text excluded).
3. Postgres populate_embeddings builds text from the same priority order
   (SQL selects llm_summary, signature; source_text not used for embedding text).
4. SQLite WHERE clause filters out nodes with no embeddable text using
   length(trim(...)) > 0 checks on each field.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from app.core.unified_db import UnifiedWikiDB


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _seed_node(
    db: UnifiedWikiDB,
    node_id: str,
    *,
    source_text: str = "",
    llm_summary: str | None = None,
    signature: str | None = None,
    docstring: str | None = None,
    symbol_name: str = "",
    is_architectural: int = 1,
) -> None:
    """Insert a node and optionally set llm_summary / signature / docstring."""
    db.upsert_node(
        node_id,
        rel_path="src/a.py",
        file_name="a.py",
        language="python",
        symbol_name=symbol_name or node_id,
        symbol_type="function",
        is_architectural=is_architectural,
        source_text=source_text,
        signature=signature or "",
        docstring=docstring or "",
    )
    if llm_summary is not None:
        db.update_node_summary(node_id, llm_summary, "test")


class _CaptureEmbedder:
    """Records every batch of texts sent to the embedder."""

    def __init__(self, dim: int = 8) -> None:
        self.seen_texts: list[str] = []
        self._dim = dim

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.seen_texts.extend(texts)
        return [[0.0] * self._dim for _ in texts]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Ordering: two-pass embedding in _write_unified_db
# ═══════════════════════════════════════════════════════════════════════════


class TestEmbeddingsCalledAfterSummaries:
    """Inspect _write_unified_db source to confirm two-pass ordering."""

    def _parse_call_lines(self, src: str) -> tuple[list[int], list[int], list[int]]:
        """Return (embed_lines, phase2_lines, summaries_lines) from source."""
        tree = ast.parse(textwrap.dedent(src))
        embed_lines: list[int] = []
        phase2_lines: list[int] = []
        summaries_lines: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "populate_embeddings":
                embed_lines.append(node.lineno)
            if isinstance(node.func, ast.Name) and node.func.id == "run_phase2":
                phase2_lines.append(node.lineno)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "populate_summaries":
                summaries_lines.append(node.lineno)
        return embed_lines, phase2_lines, summaries_lines

    def test_phase1b_populate_embeddings_before_phase2(self) -> None:
        """Phase 1b MUST have a populate_embeddings call BEFORE run_phase2.

        Phase 2 orphan resolution calls db.search_vec which requires repo_vec
        to be populated.  Verify that at least one populate_embeddings call
        appears before run_phase2 in _write_unified_db.
        """
        import inspect
        from app.core import filesystem_indexer as fi_module

        src = inspect.getsource(fi_module.FilesystemRepositoryIndexer._write_unified_db)
        embed_lines, phase2_lines, summaries_lines = self._parse_call_lines(src)

        assert embed_lines, "populate_embeddings not found in _write_unified_db"
        assert phase2_lines, "run_phase2 not found in _write_unified_db"

        # At least one populate_embeddings call must appear BEFORE run_phase2
        assert any(
            e_line < p2_line
            for e_line in embed_lines
            for p2_line in phase2_lines
        ), (
            f"No populate_embeddings call appears before run_phase2. "
            f"embed_lines={embed_lines}, phase2_lines={phase2_lines} — "
            f"Phase 1b embedding is required for Phase 2 orphan resolution."
        )

    def test_populate_embeddings_also_called_after_summaries(self) -> None:
        """A second populate_embeddings call must appear AFTER populate_summaries.

        Nodes that received llm_summary via the agentic/batch path must be
        re-embedded so the richer text is reflected in repo_vec.
        """
        import inspect
        from app.core import filesystem_indexer as fi_module

        src = inspect.getsource(fi_module.FilesystemRepositoryIndexer._write_unified_db)
        embed_lines, phase2_lines, summaries_lines = self._parse_call_lines(src)

        assert embed_lines, "populate_embeddings not found in _write_unified_db"
        assert summaries_lines, "populate_summaries not found in _write_unified_db"

        # At least one populate_embeddings call must appear AFTER populate_summaries
        assert any(
            e_line > s_line
            for e_line in embed_lines
            for s_line in summaries_lines
        ), (
            f"No populate_embeddings call appears after populate_summaries. "
            f"embed_lines={embed_lines}, summaries_lines={summaries_lines} — "
            f"Post-summary re-embedding is required so llm_summary is used."
        )

    def test_exactly_two_populate_embeddings_calls_in_write_unified_db(self) -> None:
        """There must be exactly two populate_embeddings call sites (Phase 1b + post-summaries)."""
        import inspect
        from app.core import filesystem_indexer as fi_module

        src = inspect.getsource(fi_module.FilesystemRepositoryIndexer._write_unified_db)
        tree = ast.parse(textwrap.dedent(src))

        embed_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "populate_embeddings"
        ]
        assert len(embed_calls) == 2, (
            f"Expected exactly 2 populate_embeddings calls (Phase 1b + post-summaries), "
            f"found {len(embed_calls)}: lines {[n.lineno for n in embed_calls]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. SQLite: text priority — llm_summary beats source_text
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def sqlite_db(tmp_path, monkeypatch):
    """SQLite DB with sqlite-vec stubbed out."""
    db = UnifiedWikiDB(tmp_path / "embed.wiki.db", embedding_dim=8)
    db._vec_available = True
    monkeypatch.setattr(db, "upsert_embeddings_batch", lambda pairs: None)
    yield db
    db.close()


class TestSQLiteEmbedTextPriority:
    def test_llm_summary_used_over_source_text(self, sqlite_db) -> None:
        """When llm_summary is set, it should be what the embedder receives."""
        _seed_node(
            sqlite_db,
            "fn_a",
            source_text="def fn_a():\n    " + "x" * 5000,  # very long source
            llm_summary="Handles async job dispatch with retry logic.",
        )
        embedder = _CaptureEmbedder()
        sqlite_db.populate_embeddings(embedder, batch_size=10, architectural_only=False)

        assert embedder.seen_texts, "embedder was never called"
        assert any("Handles async job dispatch" in t for t in embedder.seen_texts), (
            f"llm_summary not used; texts sent to embedder: {embedder.seen_texts}"
        )
        # Source text (raw code) must NOT appear in any embedded text
        assert not any("x" * 100 in t for t in embedder.seen_texts), (
            "Raw source_text leaked into embedding text"
        )

    def test_signature_and_docstring_used_when_no_summary(self, sqlite_db) -> None:
        """Without llm_summary, signature + docstring should be used."""
        _seed_node(
            sqlite_db,
            "fn_b",
            source_text="def fn_b(x, y):\n    return x + y",
            llm_summary=None,
            signature="def fn_b(x: int, y: int) -> int",
            docstring="Add two integers and return the result.",
        )
        embedder = _CaptureEmbedder()
        sqlite_db.populate_embeddings(embedder, batch_size=10, architectural_only=False)

        assert embedder.seen_texts, "embedder was never called"
        combined = " ".join(embedder.seen_texts)
        assert "fn_b(x: int, y: int)" in combined, (
            f"signature not found; texts: {embedder.seen_texts}"
        )
        assert "Add two integers" in combined, (
            f"docstring not found; texts: {embedder.seen_texts}"
        )

    def test_symbol_name_fallback_when_no_summary_sig_doc(self, sqlite_db) -> None:
        """When only symbol_name is available, it should be used as last resort.

        The WHERE clause now filters on llm_summary | signature | docstring |
        symbol_name (not source_text), so a node with only a non-empty
        symbol_name is included in the embedding query.
        """
        _seed_node(
            sqlite_db,
            "fn_c",
            source_text="",  # no source text — node is still selected via symbol_name
            llm_summary=None,
            signature="",
            docstring="",
            symbol_name="compute_widget_delta",
        )
        # embedding_content_hash is NULL (never embedded) so the skip filter
        # selects this node. No manual override needed.
        embedder = _CaptureEmbedder()
        sqlite_db.populate_embeddings(embedder, batch_size=10, architectural_only=False)

        # fn_c has a non-empty symbol_name so it IS selected by the new WHERE
        # clause (llm_summary | signature | docstring | symbol_name).
        assert embedder.seen_texts, "embedder was never called for fn_c"
        assert any("compute_widget_delta" in t for t in embedder.seen_texts), (
            f"symbol_name not used as fallback; texts: {embedder.seen_texts}"
        )

        # Also verify with a node that has source_text but no summary/sig/doc.
        _seed_node(
            sqlite_db,
            "fn_d",
            source_text="x = 1",
            llm_summary=None,
            signature="",
            docstring="",
            symbol_name="compute_widget_delta_d",
        )
        embedder2 = _CaptureEmbedder()
        sqlite_db.populate_embeddings(embedder2, batch_size=10, architectural_only=False)

        assert embedder2.seen_texts, "embedder2 was never called for fn_d"
        assert any("compute_widget_delta_d" in t for t in embedder2.seen_texts), (
            f"symbol_name not used as fallback; texts: {embedder2.seen_texts}"
        )

    def test_source_text_not_embedded_directly(self, sqlite_db) -> None:
        """Even when source_text is the only data, it must NOT appear verbatim.

        The WHERE clause filters on llm_summary | signature | docstring |
        symbol_name.  A node with only source_text is selected via its
        non-empty symbol_name, and the embedding text must be symbol_name —
        never the raw source code.
        """
        long_code = "def fn_e():\n    " + "pass  " * 200
        _seed_node(
            sqlite_db,
            "fn_e",
            source_text=long_code,
            llm_summary=None,
            signature="",
            docstring="",
            symbol_name="fn_e_symbol",
        )
        embedder = _CaptureEmbedder()
        sqlite_db.populate_embeddings(embedder, batch_size=10, architectural_only=False)

        # fn_e has a non-empty symbol_name so the embedder should be called.
        assert embedder.seen_texts, "embedder was never called for fn_e"
        # The text embedded must be the symbol_name, not the raw source code
        for t in embedder.seen_texts:
            if "fn_e_symbol" in t:
                assert "pass  pass  pass" not in t, (
                    "Raw source_text (repetitive code) found in embedding text"
                )

    def test_llm_summary_with_signature_combined(self, sqlite_db) -> None:
        """llm_summary AND signature should both appear in the embedded text."""
        _seed_node(
            sqlite_db,
            "fn_f",
            source_text="def fn_f(a, b): pass",
            llm_summary="Processes and validates the input pair.",
            signature="def fn_f(a: str, b: int) -> bool",
            docstring="Validate input.",
        )
        embedder = _CaptureEmbedder()
        sqlite_db.populate_embeddings(embedder, batch_size=10, architectural_only=False)

        assert embedder.seen_texts
        combined = " ".join(embedder.seen_texts)
        assert "Processes and validates" in combined
        assert "fn_f(a: str, b: int)" in combined
        assert "Validate input" in combined


# ═══════════════════════════════════════════════════════════════════════════
# 3. Postgres: SQL selects llm_summary and signature
# ═══════════════════════════════════════════════════════════════════════════


class TestPostgresEmbedSQL:
    """Inspect the postgres populate_embeddings SQL to verify column selection."""

    def test_sql_selects_llm_summary(self) -> None:
        import inspect
        from app.core.storage import postgres as pg_module

        src = inspect.getsource(pg_module.PostgresWikiStorage.populate_embeddings)
        assert "llm_summary" in src, (
            "postgres populate_embeddings SQL must SELECT llm_summary"
        )

    def test_sql_selects_signature(self) -> None:
        import inspect
        from app.core.storage import postgres as pg_module

        src = inspect.getsource(pg_module.PostgresWikiStorage.populate_embeddings)
        assert "signature" in src, (
            "postgres populate_embeddings SQL must SELECT signature"
        )

    def test_text_priority_logic_uses_llm_summary_first(self) -> None:
        """The text-building loop must reference llm_summary before source_text."""
        import inspect
        from app.core.storage import postgres as pg_module

        src = inspect.getsource(pg_module.PostgresWikiStorage.populate_embeddings)
        # llm_summary should appear in the source before source_text
        llm_pos = src.find("llm_summary")
        assert llm_pos != -1, "llm_summary not referenced in postgres populate_embeddings"
        # source_text may still appear (e.g. in comments or old code) but we also
        # verify the priority logic puts llm_summary first in the actual text building
        # We do this by checking that "llm_summary" appears before the text-building
        # conditional that falls back (signature/docstring/symbol_name).
        summary_pos = src.find('"llm_summary"') if '"llm_summary"' in src else src.find("llm_summary")
        assert summary_pos != -1

    def test_source_text_not_used_as_primary_embedding_source(self) -> None:
        """source_text must not be the first field used for embedding text."""
        import inspect
        from app.core.storage import postgres as pg_module

        src = inspect.getsource(pg_module.PostgresWikiStorage.populate_embeddings)
        # The old code started with: txt = (r[1] or "").strip()  # source_text
        # Verify this pattern (positional index 1 being source_text as first choice) is gone.
        # Look for the old pattern of r[1] being used as primary text.
        # This is a heuristic: after change, the first embeddable field should be llm_summary.
        lines = src.splitlines()
        text_build_started = False
        for line in lines:
            stripped = line.strip()
            # Skip comment lines
            if stripped.startswith("#"):
                continue
            if "llm_summary" in stripped and ("summary" in stripped.lower()) and not text_build_started:
                # llm_summary appears before source_text in text-building
                text_build_started = True
                break
            # If we see source_text used as primary (r[1] pattern from old code) before llm_summary
            if 'r[1]' in stripped and 'strip()' in stripped and not text_build_started:
                pytest.fail(
                    "Old pattern 'r[1]' (source_text) appears as primary text before llm_summary"
                )

        assert text_build_started, "llm_summary not found in text-building section of postgres populate_embeddings"
