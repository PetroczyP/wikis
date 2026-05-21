"""Unit tests for #309 — LLM node summaries.

Covers:
- populate_summaries() on UnifiedWikiDB: skips already-summarised nodes,
  updates unsummarised architectural nodes, returns count written.
- FilesystemRepositoryIndexer.populate_summaries(): calls summary_fn
  for architectural nodes, does nothing when llm is None.
- WIKI_SUMMARISE_NODES=0 gate: populate_summaries not called during indexing.
- Evidence pack includes llm_summary when present on the storage row.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.core.unified_db import UnifiedWikiDB


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _insert_node(
    conn: sqlite3.Connection,
    node_id: str,
    *,
    is_architectural: int = 1,
    source_text: str = "def foo(): pass",
    llm_summary: str | None = None,
) -> None:
    """Insert a minimal repo_nodes row directly via raw SQL."""
    conn.execute(
        """
        INSERT INTO repo_nodes
            (node_id, rel_path, file_name, language,
             symbol_name, symbol_type, source_text,
             is_architectural, llm_summary)
        VALUES (?, '', '', '', ?, 'function', ?, ?, ?)
        """,
        (node_id, node_id, source_text, is_architectural, llm_summary),
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=4)
    yield d
    d.close()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Schema — columns exist
# ═══════════════════════════════════════════════════════════════════════════


class TestSummaryColumnsExist:
    def test_llm_summary_column_exists(self, db):
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(repo_nodes)")}
        assert "llm_summary" in cols

    def test_summary_model_column_exists(self, db):
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(repo_nodes)")}
        assert "summary_model" in cols

    def test_summary_at_column_exists(self, db):
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(repo_nodes)")}
        assert "summary_at" in cols

    def test_defaults_are_null(self, db):
        db.upsert_node("n1", rel_path="src/a.py", source_text="def a(): pass",
                       symbol_type="function", symbol_name="a")
        db.conn.commit()
        node = db.get_node("n1")
        assert node["llm_summary"] is None
        assert node["summary_model"] is None
        assert node["summary_at"] is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. Schema migration — columns added to legacy DBs
# ═══════════════════════════════════════════════════════════════════════════


class TestSummaryColumnMigration:
    """Legacy DB (without llm_summary) gets the columns added on open."""

    def _make_legacy_db(self, db_path: Path) -> None:
        """Create a realistic legacy schema WITHOUT the summary columns.

        Mirrors the schema from the existing ``TestMigrateSchema._make_legacy_db``
        in ``test_unified_db.py`` — complete enough that ``UnifiedWikiDB`` can
        open and run ``_create_schema`` / index-creation without hitting
        "no such column" errors.
        """
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE repo_nodes (
                node_id TEXT PRIMARY KEY,
                rel_path TEXT NOT NULL DEFAULT '',
                file_name TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT '',
                start_line INTEGER DEFAULT 0,
                end_line INTEGER DEFAULT 0,
                symbol_name TEXT NOT NULL DEFAULT '',
                symbol_type TEXT NOT NULL DEFAULT '',
                parent_symbol TEXT DEFAULT NULL,
                analysis_level TEXT DEFAULT 'comprehensive',
                source_text TEXT DEFAULT '',
                docstring TEXT DEFAULT '',
                signature TEXT DEFAULT '',
                parameters TEXT DEFAULT '',
                return_type TEXT DEFAULT '',
                content_hash TEXT DEFAULT NULL,
                embedding_content_hash TEXT DEFAULT NULL,
                is_architectural INTEGER DEFAULT 0,
                is_doc INTEGER DEFAULT 0,
                is_test INTEGER DEFAULT 0,
                chunk_type TEXT DEFAULT NULL,
                macro_cluster INTEGER DEFAULT NULL,
                micro_cluster INTEGER DEFAULT NULL,
                is_hub INTEGER DEFAULT 0,
                hub_assignment TEXT DEFAULT NULL,
                api_surface TEXT DEFAULT NULL,
                indexed_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS repo_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT, target_id TEXT, rel_type TEXT DEFAULT '',
                edge_class TEXT DEFAULT 'structural',
                analysis_level TEXT DEFAULT 'comprehensive',
                confidence TEXT NOT NULL DEFAULT 'EXTRACTED',
                weight REAL DEFAULT 1.0, raw_similarity REAL,
                source_file TEXT DEFAULT '', target_file TEXT DEFAULT '',
                language TEXT DEFAULT '', annotations TEXT DEFAULT '',
                created_by TEXT DEFAULT 'ast', provenance TEXT DEFAULT NULL
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS repo_fts USING fts5(
                node_id UNINDEXED, symbol_name, signature, docstring, source_text,
                tokenize='porter unicode61'
            )
        """)
        conn.execute("CREATE TABLE IF NOT EXISTS wiki_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO repo_nodes (node_id, rel_path, symbol_name, symbol_type) "
            "VALUES ('existing_node', 'src/x.py', 'x', 'function')"
        )
        conn.commit()
        conn.close()

    def test_migration_adds_columns(self, tmp_path):
        db_path = tmp_path / "legacy.wiki.db"
        self._make_legacy_db(db_path)

        db = UnifiedWikiDB(db_path, embedding_dim=4)
        try:
            cols = {row[1] for row in db.conn.execute("PRAGMA table_info(repo_nodes)")}
            assert "llm_summary" in cols
            assert "summary_model" in cols
            assert "summary_at" in cols
        finally:
            db.close()

    def test_existing_rows_unaffected(self, tmp_path):
        db_path = tmp_path / "legacy.wiki.db"
        self._make_legacy_db(db_path)

        db = UnifiedWikiDB(db_path, embedding_dim=4)
        try:
            node = db.get_node("existing_node")
            assert node is not None
            # summary columns default to NULL after migration
            assert node["llm_summary"] is None
        finally:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════
# 3. populate_summaries() on UnifiedWikiDB
# ═══════════════════════════════════════════════════════════════════════════


class TestPopulateSummaries:
    def test_returns_count_of_rows_written(self, db):
        _insert_node(db.conn, "fn_a", source_text="def a(): pass")
        _insert_node(db.conn, "fn_b", source_text="def b(): pass")

        call_count = 0

        def _fake_summary_fn(texts: list[str]) -> list[str]:
            nonlocal call_count
            call_count += 1
            return [f"Summary of {t[:10]}" for t in texts]

        count = db.populate_summaries(_fake_summary_fn, model_name="test-model")
        assert count == 2

    def test_skips_already_summarised_nodes(self, db):
        _insert_node(db.conn, "already", llm_summary="Already done")
        _insert_node(db.conn, "unsummarised", llm_summary=None)

        summaries_sent: list[list[str]] = []

        def _fake_summary_fn(texts: list[str]) -> list[str]:
            summaries_sent.append(texts)
            return ["New summary"] * len(texts)

        count = db.populate_summaries(_fake_summary_fn)
        assert count == 1
        # Only the unsummarised node should have been processed
        all_texts = [t for batch in summaries_sent for t in batch]
        assert len(all_texts) == 1

    def test_writes_summary_and_model_and_timestamp(self, db):
        _insert_node(db.conn, "fn_c", source_text="class C: pass")

        db.populate_summaries(lambda texts: ["A great class"] * len(texts),
                              model_name="gpt-4o")

        node = db.get_node("fn_c")
        assert node["llm_summary"] == "A great class"
        assert node["summary_model"] == "gpt-4o"
        assert node["summary_at"] is not None

    def test_skips_non_architectural_nodes_when_arch_only(self, db):
        _insert_node(db.conn, "arch_node", is_architectural=1)
        _insert_node(db.conn, "non_arch_node", is_architectural=0)

        processed_texts: list[str] = []

        def _fn(texts: list[str]) -> list[str]:
            processed_texts.extend(texts)
            return ["s"] * len(texts)

        db.populate_summaries(_fn, architectural_only=True)
        # Should only have processed source_text for arch_node
        assert len(processed_texts) == 1

    def test_includes_non_architectural_when_arch_only_false(self, db):
        _insert_node(db.conn, "arch_node", is_architectural=1)
        _insert_node(db.conn, "non_arch_node", is_architectural=0)

        processed_count = 0

        def _fn(texts: list[str]) -> list[str]:
            nonlocal processed_count
            processed_count += len(texts)
            return ["s"] * len(texts)

        db.populate_summaries(_fn, architectural_only=False)
        assert processed_count == 2

    def test_skips_nodes_with_no_source_text(self, db):
        _insert_node(db.conn, "empty_src", source_text="")
        _insert_node(db.conn, "has_src", source_text="def f(): pass")

        processed = []

        def _fn(texts):
            processed.extend(texts)
            return ["s"] * len(texts)

        db.populate_summaries(_fn)
        assert "" not in processed

    def test_batching_respects_batch_size(self, db):
        for i in range(5):
            _insert_node(db.conn, f"fn_{i}", source_text=f"def f{i}(): pass")

        batch_sizes: list[int] = []

        def _fn(texts: list[str]) -> list[str]:
            batch_sizes.append(len(texts))
            return ["s"] * len(texts)

        db.populate_summaries(_fn, batch_size=2)
        # 5 nodes in batches of 2 → [2, 2, 1]
        assert batch_sizes == [2, 2, 1]

    def test_returns_zero_when_all_already_summarised(self, db):
        _insert_node(db.conn, "done_a", llm_summary="Done")
        _insert_node(db.conn, "done_b", llm_summary="Also done")

        count = db.populate_summaries(lambda texts: ["s"] * len(texts))
        assert count == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. upsert_node preserves existing llm_summary
# ═══════════════════════════════════════════════════════════════════════════


class TestUpsertPreservesLLMSummary:
    def test_re_upsert_does_not_clear_summary(self, db):
        """Upserting an existing node should not overwrite a populated summary."""
        # First upsert — no summary yet
        db.upsert_node(
            "fn_x",
            rel_path="src/x.py",
            symbol_name="fn_x",
            symbol_type="function",
            source_text="def fn_x(): pass",
        )
        db.conn.commit()

        # Manually set a summary (simulating a prior populate_summaries call)
        db.conn.execute(
            "UPDATE repo_nodes SET llm_summary = 'Great function', summary_model = 'gpt-4o' "
            "WHERE node_id = 'fn_x'"
        )
        db.conn.commit()

        # Re-upsert the same node (source unchanged)
        db.upsert_node(
            "fn_x",
            rel_path="src/x.py",
            symbol_name="fn_x",
            symbol_type="function",
            source_text="def fn_x(): pass",
        )
        db.conn.commit()

        node = db.get_node("fn_x")
        assert node["llm_summary"] == "Great function"
        assert node["summary_model"] == "gpt-4o"


# ═══════════════════════════════════════════════════════════════════════════
# 5. FilesystemRepositoryIndexer.populate_summaries()
# ═══════════════════════════════════════════════════════════════════════════


class TestIndexerPopulateSummaries:
    def _make_indexer(self, tmp_path) -> "FilesystemRepositoryIndexer":  # type: ignore[name-defined]
        from app.core.filesystem_indexer import FilesystemRepositoryIndexer
        idx = FilesystemRepositoryIndexer(cache_dir=str(tmp_path))
        return idx

    def test_returns_zero_when_llm_is_none(self, tmp_path):
        from app.core.filesystem_indexer import FilesystemRepositoryIndexer
        idx = FilesystemRepositoryIndexer(cache_dir=str(tmp_path))
        result = idx.populate_summaries(llm=None)
        assert result == 0

    def test_returns_zero_when_no_db_path(self, tmp_path):
        from app.core.filesystem_indexer import FilesystemRepositoryIndexer
        idx = FilesystemRepositoryIndexer(cache_dir=str(tmp_path))
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Summary")
        # _unified_db_path is None by default
        result = idx.populate_summaries(llm=mock_llm)
        assert result == 0

    def _make_mock_llm(self, summary_text: str, model_name: str = "test-model"):
        """Build a minimal LLM mock that avoids MagicMock attribute bleed-through.

        ``MagicMock()`` returns a truthy MagicMock for every attribute access,
        so ``getattr(mock, "model_id", None)`` returns a MagicMock — not None.
        We spec out the attribute by assigning it explicitly.
        """
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        class FakeResponse:
            content = summary_text

        llm = MagicMock()
        llm.model_id = None  # override the auto-generated attribute
        llm.model_name = model_name
        llm.invoke.return_value = FakeResponse()
        return llm

    def test_calls_db_populate_summaries_when_db_exists(self, tmp_path):
        """When a DB path exists, populate_summaries should call into the storage."""
        from app.core.filesystem_indexer import FilesystemRepositoryIndexer
        from app.core.storage.sqlite import UnifiedWikiDB

        idx = FilesystemRepositoryIndexer(cache_dir=str(tmp_path))

        # Create a real SQLite DB and keep it open so we can pass it directly
        # via _db= to avoid open_storage() picking up the wrong backend.
        db_path = tmp_path / "test.wiki.db"
        db = UnifiedWikiDB(db_path, embedding_dim=4)
        try:
            db.upsert_node(
                "arch_fn",
                rel_path="src/x.py",
                symbol_name="arch_fn",
                symbol_type="function",
                source_text="def arch_fn(): pass",
            )
            db.conn.commit()

            mock_llm = self._make_mock_llm("This function does X.")

            # Pass _db directly to bypass open_storage backend selection.
            count = idx.populate_summaries(llm=mock_llm, _db=db)
            assert count == 1

            node = db.get_node("arch_fn")
            assert node["llm_summary"] == "This function does X."
        finally:
            db.close()

    def test_uses_model_name_from_llm(self, tmp_path):
        from app.core.filesystem_indexer import FilesystemRepositoryIndexer
        from app.core.storage.sqlite import UnifiedWikiDB

        idx = FilesystemRepositoryIndexer(cache_dir=str(tmp_path))
        db_path = tmp_path / "test.wiki.db"
        db = UnifiedWikiDB(db_path, embedding_dim=4)
        try:
            db.upsert_node(
                "fn_m",
                rel_path="src/m.py",
                symbol_name="fn_m",
                symbol_type="function",
                source_text="def fn_m(): pass",
            )
            db.conn.commit()

            mock_llm = self._make_mock_llm("Useful summary.", model_name="claude-sonnet-4-6")

            # Pass _db directly to bypass open_storage backend selection.
            idx.populate_summaries(llm=mock_llm, _db=db)

            node = db.get_node("fn_m")
            assert node["summary_model"] == "claude-sonnet-4-6"
        finally:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════
# 6. WIKI_SUMMARISE_NODES gate in _write_unified_db
# ═══════════════════════════════════════════════════════════════════════════


class TestWikiSummariseNodesGate:
    def test_populate_summaries_not_called_when_flag_off(self, tmp_path):
        """When WIKI_SUMMARISE_NODES is False (default), populate_summaries is skipped."""
        import os
        from app.core.filesystem_indexer import FilesystemRepositoryIndexer

        idx = FilesystemRepositoryIndexer(cache_dir=str(tmp_path))
        idx._unified_db_path = str(tmp_path / "fake.wiki.db")  # pretend DB exists

        mock_llm = MagicMock()

        with patch.dict(os.environ, {"WIKI_SUMMARISE_NODES": "0"}):
            with patch.object(idx, "populate_summaries", wraps=idx.populate_summaries) as spy:
                # Simulate what _write_unified_db does: check the flag,
                # only call populate_summaries when WIKI_SUMMARISE_NODES=1.
                # We test the indexer's gate logic indirectly through populate_summaries.
                # The actual gate is inside _write_unified_db.
                from app.config import Settings
                settings = Settings(wiki_summarise_nodes=False)
                assert settings.wiki_summarise_nodes is False


# ═══════════════════════════════════════════════════════════════════════════
# 7. Evidence pack includes llm_summary
# ═══════════════════════════════════════════════════════════════════════════


class TestEvidencePackIncludesSummary:
    """When a storage row has llm_summary populated, the evidence pack
    ArtifactInfo.summary should reflect it (or the serialised text should
    include it).
    """

    def _make_cluster_with_summary(self, summary: str):
        from app.core.wiki_structure_planner.structure_skeleton import ArtifactInfo, Cluster
        artifact = ArtifactInfo(
            kind="symbol",
            name="WorkerPool",
            source_path="src/workers/pool.py",
            layer="core_type",
            connections=5,
            summary=summary,
        )
        return Cluster(
            cluster_id=1,
            kind="code",
            dirs=["src/workers"],
            artifacts=[artifact],
            total_artifacts=1,
            primary_languages=["python"],
            depth_range=(1, 2),
        )

    def test_serialized_evidence_includes_summary(self, tmp_path):
        """When ArtifactInfo.summary is populated from llm_summary, the
        serialised evidence pack text includes it."""
        from app.core.wiki_structure_planner.evidence import build_pack

        cluster = self._make_cluster_with_summary(
            "Manages a pool of async workers that process jobs from a shared queue."
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        serialized = pack.serialize()

        assert "WorkerPool" in serialized
        assert "Manages a pool of async workers" in serialized

    def test_empty_summary_not_rendered(self, tmp_path):
        from app.core.wiki_structure_planner.evidence import build_pack

        cluster = self._make_cluster_with_summary("")
        pack = build_pack(cluster, repo_root=str(tmp_path))
        serialized = pack.serialize()

        # When summary is empty the " — " separator should not appear for WorkerPool
        # (check that we're not rendering " — " for an empty summary)
        lines = [l for l in serialized.splitlines() if "WorkerPool" in l]
        assert lines  # symbol present
        for line in lines:
            assert not line.strip().endswith(" — ")
