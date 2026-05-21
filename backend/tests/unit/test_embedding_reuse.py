"""Tests for #116 PR 4 embedding reuse — populate_embeddings must skip
nodes whose embedding text (llm_summary | signature | docstring | symbol_name)
hasn't changed since the last embed.

The contract:

* First call embeds every architectural node, stamps embedding_content_hash
  as hash(embedding_text).
* Second call with no changes embeds zero nodes (no calls to embedding_fn).
* After mutating a node's signature or llm_summary (i.e. the actual embedding
  text), only that node gets re-embedded on the next call.
* Changing source_text alone does NOT trigger re-embedding because source_text
  is not part of the embedding text.
* Pre-PR4 rows with NULL embedding_content_hash are re-embedded once on
  the next populate_embeddings, then stamped.

embedding_text_hash DB-level filter contract (new):

* upsert_node computes embedding_text_hash = sha256(emb_text) and persists it.
* populate_embeddings SQL WHERE clause selects only rows where
  embedding_content_hash IS NULL OR embedding_content_hash != embedding_text_hash.
* After embedding, embedding_content_hash is stamped to match embedding_text_hash.
* update_node_summary recomputes embedding_text_hash with the new llm_summary,
  causing the node to be selected on the next populate_embeddings call.
"""

from __future__ import annotations

import pytest

from app.core.unified_db import UnifiedWikiDB


@pytest.fixture()
def db(tmp_path, monkeypatch):
    # The skip-filter we care about runs inside the SELECT — independent
    # of whether sqlite-vec is loadable. To keep these tests portable we
    # pretend vec is available and replace upsert_embeddings_batch with a
    # no-op so populate_embeddings runs end-to-end without writing real
    # vectors. The test gates on which texts the embedder *sees*, not on
    # repo_vec contents.
    d = UnifiedWikiDB(tmp_path / "embed.wiki.db", embedding_dim=8)
    d._vec_available = True
    monkeypatch.setattr(d, "upsert_embeddings_batch", lambda pairs: None)
    yield d
    d.close()


def _seed(db: UnifiedWikiDB, node_id: str, source: str) -> None:
    db.upsert_node(
        node_id,
        rel_path="x.py",
        file_name="x.py",
        language="python",
        symbol_name=node_id,
        symbol_type="function",
        is_architectural=1,
        source_text=source,
    )


class _CountingEmbedder:
    """Records every (text→vector) call so tests can assert no-op behavior."""

    def __init__(self, dim: int = 8) -> None:
        self.calls: list[list[str]] = []
        self._dim = dim

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0] * self._dim for _ in texts]

    @property
    def total_texts(self) -> int:
        return sum(len(c) for c in self.calls)


def _read_embedding_hashes(db: UnifiedWikiDB) -> dict[str, str | None]:
    rows = db.conn.execute(
        "SELECT node_id, embedding_content_hash FROM repo_nodes"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingReuse:
    def test_first_run_embeds_every_node_and_stamps_hash(
        self, db: UnifiedWikiDB,
    ) -> None:
        _seed(db, "a", "def a(): pass")
        _seed(db, "b", "def b(): pass")
        embedder = _CountingEmbedder()

        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 2
        assert embedder.total_texts == 2

        hashes = _read_embedding_hashes(db)
        # Every node has a non-NULL embedding_content_hash now.
        assert hashes["a"] is not None
        assert hashes["b"] is not None

    def test_second_run_with_no_changes_is_a_noop(
        self, db: UnifiedWikiDB,
    ) -> None:
        _seed(db, "a", "def a(): pass")
        _seed(db, "b", "def b(): pass")

        # First run — embed everything.
        db.populate_embeddings(_CountingEmbedder(), batch_size=10)

        # Second run with a fresh embedder — should be called zero times.
        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 0
        assert embedder.total_texts == 0
        assert embedder.calls == []

    def test_modified_signature_causes_reembed_unchanged_node_skipped(
        self, db: UnifiedWikiDB,
    ) -> None:
        """Changing embedding text (signature) triggers re-embed; source_text change does not."""
        _seed(db, "a", "def a(): pass")
        _seed(db, "b", "def b(): pass")

        db.populate_embeddings(_CountingEmbedder(), batch_size=10)

        # Change only source_text for "a" — embedding text (symbol_name="a") is unchanged.
        _seed(db, "a", "def a(): return 1")
        embedder_src_only = _CountingEmbedder()
        n_src_only = db.populate_embeddings(embedder_src_only, batch_size=10)
        # source_text is not part of embedding text → no re-embed.
        assert n_src_only == 0, (
            "Changing source_text alone must not trigger re-embedding "
            "(embedding text = llm_summary | signature | docstring | symbol_name)"
        )
        assert embedder_src_only.calls == []

        # Now change "a"'s signature via upsert_node so embedding_text_hash is
        # recomputed correctly — raw SQL would bypass the hash update logic.
        db.upsert_node(
            "a",
            rel_path="x.py",
            file_name="x.py",
            language="python",
            symbol_name="a",
            symbol_type="function",
            is_architectural=1,
            source_text="def a(): pass",
            signature="def a(x: int) -> int",
        )

        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 1, f"Expected 1 re-embedded node (signature changed), got {n}"
        assert embedder.total_texts == 1
        # "a" was re-embedded with the new signature; "b" was skipped.
        combined = " ".join(t for batch in embedder.calls for t in batch)
        assert "def a(x: int) -> int" in combined, (
            f"New signature not in embedding text: {embedder.calls}"
        )

    def test_source_text_change_alone_does_not_trigger_reembed(
        self, db: UnifiedWikiDB,
    ) -> None:
        """source_text is excluded from embedding text; changing it must not cause re-embed."""
        _seed(db, "a", "def a(): pass")
        db.populate_embeddings(_CountingEmbedder(), batch_size=10)

        # Mutate source_text — embedding text (symbol_name="a") is unchanged.
        _seed(db, "a", "def a():\n    return 42  # changed")

        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 0, (
            "source_text-only change must not cause re-embedding; "
            f"embedding_fn called with: {embedder.calls}"
        )
        assert embedder.calls == []

    def test_pre_pr4_null_embedding_hash_is_picked_up_then_stamped(
        self, db: UnifiedWikiDB,
    ) -> None:
        _seed(db, "a", "def a(): pass")

        # Simulate a pre-PR4 row: clear embedding_content_hash.
        db.conn.execute(
            "UPDATE repo_nodes SET embedding_content_hash = NULL WHERE node_id = ?",
            ("a",),
        )
        db.conn.commit()
        assert _read_embedding_hashes(db)["a"] is None

        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 1
        # Now stamped — next run is a no-op.
        assert _read_embedding_hashes(db)["a"] is not None
        n2 = db.populate_embeddings(_CountingEmbedder(), batch_size=10)
        assert n2 == 0


class TestEmbeddingTextHash:
    """Validate the embedding_text_hash DB-level freshness filter."""

    def test_upsert_node_sets_embedding_text_hash(self, db: UnifiedWikiDB) -> None:
        """upsert_node must persist embedding_text_hash for nodes with embeddable text."""
        _seed(db, "a", "def a(): pass")
        row = db.conn.execute(
            "SELECT embedding_text_hash FROM repo_nodes WHERE node_id = ?", ("a",)
        ).fetchone()
        assert row is not None
        # symbol_name="a" is the embeddable text → hash must be set
        assert row["embedding_text_hash"] is not None, (
            "embedding_text_hash must be populated by upsert_node"
        )

    def test_embedding_content_hash_stamped_to_match_text_hash(
        self, db: UnifiedWikiDB
    ) -> None:
        """After populate_embeddings, embedding_content_hash == embedding_text_hash."""
        _seed(db, "a", "def a(): pass")
        db.populate_embeddings(_CountingEmbedder(), batch_size=10)
        row = db.conn.execute(
            "SELECT embedding_content_hash, embedding_text_hash FROM repo_nodes WHERE node_id = ?",
            ("a",),
        ).fetchone()
        assert row["embedding_content_hash"] == row["embedding_text_hash"], (
            "embedding_content_hash must equal embedding_text_hash after embedding"
        )

    def test_second_run_no_full_table_scan(self, db: UnifiedWikiDB) -> None:
        """Second populate_embeddings call fetches 0 rows from DB (DB-level filter)."""
        _seed(db, "a", "def a(): pass")
        _seed(db, "b", "def b(): pass")
        db.populate_embeddings(_CountingEmbedder(), batch_size=10)

        # Intercept the SQL query count — no rows should be fetched from DB.
        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 0
        assert embedder.calls == [], "embedder must not be called when nothing changed"

    def test_upsert_with_changed_signature_updates_embedding_text_hash(
        self, db: UnifiedWikiDB
    ) -> None:
        """Re-upserting a node with a different signature must update embedding_text_hash."""
        _seed(db, "a", "def a(): pass")
        row_before = db.conn.execute(
            "SELECT embedding_text_hash FROM repo_nodes WHERE node_id = ?", ("a",)
        ).fetchone()
        old_hash = row_before["embedding_text_hash"]

        # Re-upsert with different signature → embedding text changes
        db.upsert_node(
            "a",
            rel_path="x.py",
            file_name="x.py",
            language="python",
            symbol_name="a",
            symbol_type="function",
            is_architectural=1,
            source_text="def a(x: int): pass",
            signature="def a(x: int) -> None",
        )
        row_after = db.conn.execute(
            "SELECT embedding_text_hash FROM repo_nodes WHERE node_id = ?", ("a",)
        ).fetchone()
        assert row_after["embedding_text_hash"] != old_hash, (
            "embedding_text_hash must change when signature changes"
        )

    def test_update_node_summary_updates_embedding_text_hash(
        self, db: UnifiedWikiDB
    ) -> None:
        """update_node_summary must recompute embedding_text_hash with the new summary."""
        _seed(db, "a", "def a(): pass")
        row_before = db.conn.execute(
            "SELECT embedding_text_hash FROM repo_nodes WHERE node_id = ?", ("a",)
        ).fetchone()
        old_hash = row_before["embedding_text_hash"]

        # Stamp a fresh summary → embedding text changes
        db.update_node_summary("a", "Handles async dispatch logic.", model_name="test-model")

        row_after = db.conn.execute(
            "SELECT embedding_text_hash FROM repo_nodes WHERE node_id = ?", ("a",)
        ).fetchone()
        assert row_after["embedding_text_hash"] != old_hash, (
            "embedding_text_hash must change when llm_summary is updated"
        )

    def test_update_node_summary_causes_reembed_on_next_call(
        self, db: UnifiedWikiDB
    ) -> None:
        """A node that receives a new llm_summary must be re-embedded next run."""
        _seed(db, "a", "def a(): pass")
        # First embed — stamps embedding_content_hash
        db.populate_embeddings(_CountingEmbedder(), batch_size=10)

        # Update summary — changes embedding_text_hash
        db.update_node_summary("a", "New LLM summary text for node a.", model_name="m")

        # Next embed must pick up node "a" again
        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 1, f"Expected 1 re-embed after summary update, got {n}"
        combined = " ".join(t for batch in embedder.calls for t in batch)
        assert "New LLM summary text" in combined, (
            f"New summary not used in embedding text: {embedder.calls}"
        )

    def test_reindex_preserves_embedding_text_hash_with_stored_llm_summary(
        self, db: UnifiedWikiDB
    ) -> None:
        """Re-upserting a node must use the stored llm_summary when computing
        embedding_text_hash, not treat it as absent.

        This tests Bug 1: on re-index the incoming node dict does not carry
        llm_summary (that column is excluded from ON CONFLICT SET), so naively
        recomputing the hash without the stored summary would overwrite the
        correct hash and desync the freshness filter.
        """
        # Seed the node and stamp an llm_summary directly (simulating a prior
        # populate_summaries run).
        _seed(db, "a", "def a(): pass")
        db.update_node_summary("a", "Summarised: handles async dispatch.", model_name="m")

        # Capture the hash after summary was written.
        row_with_summary = db.conn.execute(
            "SELECT embedding_text_hash FROM repo_nodes WHERE node_id = ?", ("a",)
        ).fetchone()
        expected_hash = row_with_summary["embedding_text_hash"]
        assert expected_hash is not None

        # Re-upsert WITHOUT llm_summary (exactly what re-index does).
        db.upsert_node(
            "a",
            rel_path="x.py",
            file_name="x.py",
            language="python",
            symbol_name="a",
            symbol_type="function",
            is_architectural=1,
            source_text="def a(): pass",
            # No llm_summary key — re-index omits it.
        )

        row_after = db.conn.execute(
            "SELECT embedding_text_hash, llm_summary FROM repo_nodes WHERE node_id = ?",
            ("a",),
        ).fetchone()
        # llm_summary must be preserved (ON CONFLICT excludes it).
        assert row_after["llm_summary"] == "Summarised: handles async dispatch.", (
            "llm_summary was clobbered by re-upsert"
        )
        # embedding_text_hash must reflect the stored llm_summary, not be recomputed
        # as if llm_summary were empty.
        assert row_after["embedding_text_hash"] == expected_hash, (
            "Re-index must preserve embedding_text_hash computed with stored llm_summary; "
            f"got {row_after['embedding_text_hash']!r}, expected {expected_hash!r}"
        )

    def test_populate_summaries_updates_embedding_text_hash(
        self, db: UnifiedWikiDB
    ) -> None:
        """populate_summaries batch path must recompute embedding_text_hash with
        the new llm_summary so populate_embeddings detects the node as stale.

        This tests Bug 3: without this fix, nodes summarised by the batch path
        would have embedding_text_hash based on signature/docstring/symbol_name
        only, and the next populate_embeddings call would skip them because
        embedding_content_hash already equals the old (pre-summary) hash.
        """
        _seed(db, "a", "def a(): pass")

        # First embed — stamps embedding_content_hash to match embedding_text_hash.
        db.populate_embeddings(_CountingEmbedder(), batch_size=10)
        row_after_embed = db.conn.execute(
            "SELECT embedding_content_hash, embedding_text_hash FROM repo_nodes WHERE node_id = ?",
            ("a",),
        ).fetchone()
        assert row_after_embed["embedding_content_hash"] == row_after_embed["embedding_text_hash"]

        # Now simulate a batch populate_summaries run.
        summary_written = "Batch summary: parses widget events."
        db.populate_summaries(
            summary_fn=lambda texts: [summary_written] * len(texts),
            model_name="batch-model",
            architectural_only=False,
        )

        # embedding_text_hash must have changed (now includes llm_summary).
        row_after_summary = db.conn.execute(
            "SELECT embedding_content_hash, embedding_text_hash FROM repo_nodes WHERE node_id = ?",
            ("a",),
        ).fetchone()
        assert row_after_summary["embedding_text_hash"] != row_after_embed["embedding_text_hash"], (
            "populate_summaries must update embedding_text_hash when llm_summary is written"
        )
        # embedding_content_hash is now stale → populate_embeddings must re-embed.
        assert row_after_summary["embedding_content_hash"] != row_after_summary["embedding_text_hash"], (
            "After populate_summaries, embedding_content_hash should differ from embedding_text_hash"
            " so populate_embeddings knows to re-embed the node"
        )

        # Confirm populate_embeddings actually re-embeds and uses the summary.
        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 1, f"Expected 1 re-embed after batch summarisation, got {n}"
        combined = " ".join(t for batch in embedder.calls for t in batch)
        assert "Batch summary" in combined, (
            f"llm_summary not in embedding text after batch populate_summaries: {embedder.calls}"
        )
