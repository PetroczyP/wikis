"""Regression tests for PR #344 review fixes.

Fix 1 — shared _build_embedding_text: hash computed at upsert time must
         match the text built at embed time; symbol_name is last-resort only.

Fix 2 — SQLite IN-clause chunking: batches > 999 node_ids no longer crash.

Fix 3 — _parse_numbered_list: preamble before "1." is discarded, not counted.

Fix 4 — _batch_summarise per-node fallback logs instead of silently swallowing.
"""

from __future__ import annotations

import logging

import pytest

from app.core.unified_db import UnifiedWikiDB
from app.core.storage.sqlite import _build_embedding_text as sqlite_build
from app.core.storage.postgres import _build_embedding_text as pg_build
from app.core.filesystem_indexer import _parse_numbered_list, _batch_summarise


# ═══════════════════════════════════════════════════════════════════════════
# Fix 1 — shared _build_embedding_text correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildEmbeddingText:
    """Both backends must export identical _build_embedding_text behaviour."""

    @pytest.mark.parametrize("build_fn", [sqlite_build, pg_build])
    def test_symbol_name_excluded_when_llm_summary_present(self, build_fn) -> None:
        """symbol_name must NOT appear in embedding text when llm_summary is set.

        This is the core correctness bug: the old code joined ALL non-empty
        parts, producing 'summary | symbol' instead of 'summary'.  The
        embedder sees 'summary' (priority logic), so the hashes never matched.
        """
        txt = build_fn(
            llm_summary="Handles async dispatch with retry.",
            signature=None,
            docstring=None,
            symbol_name="my_func",
        )
        assert "Handles async dispatch" in txt
        assert "my_func" not in txt, (
            "symbol_name must be excluded when llm_summary is present"
        )

    @pytest.mark.parametrize("build_fn", [sqlite_build, pg_build])
    def test_symbol_name_used_as_last_resort(self, build_fn) -> None:
        """symbol_name IS used when llm_summary, signature, and docstring are all empty."""
        txt = build_fn(
            llm_summary=None,
            signature=None,
            docstring=None,
            symbol_name="my_func",
        )
        assert txt == "my_func"

    @pytest.mark.parametrize("build_fn", [sqlite_build, pg_build])
    def test_symbol_name_excluded_when_signature_present(self, build_fn) -> None:
        """symbol_name must be excluded when signature (but not llm_summary) is present."""
        txt = build_fn(
            llm_summary=None,
            signature="def my_func(x: int) -> bool",
            docstring=None,
            symbol_name="my_func",
        )
        assert "def my_func" in txt
        assert txt.count("my_func") == 1, (
            "symbol_name appears twice — once in signature is ok but not as a separate part"
        )
        # symbol_name should not be appended as an independent segment
        parts = txt.split(" | ")
        assert not any(p == "my_func" for p in parts), (
            "symbol_name added as a standalone segment despite signature being present"
        )

    @pytest.mark.parametrize("build_fn", [sqlite_build, pg_build])
    def test_llm_summary_and_signature_and_docstring_all_included(self, build_fn) -> None:
        txt = build_fn(
            llm_summary="Top-level dispatcher.",
            signature="def dispatch(ev: Event) -> None",
            docstring="Dispatches events to registered handlers.",
            symbol_name="dispatch",
        )
        assert "Top-level dispatcher." in txt
        assert "def dispatch" in txt
        assert "Dispatches events" in txt
        # symbol_name not appended separately
        parts = txt.split(" | ")
        assert not any(p == "dispatch" for p in parts)

    @pytest.mark.parametrize("build_fn", [sqlite_build, pg_build])
    def test_empty_string_inputs_treated_as_none(self, build_fn) -> None:
        txt = build_fn(
            llm_summary="",
            signature="",
            docstring="",
            symbol_name="fallback_sym",
        )
        assert txt == "fallback_sym"

    @pytest.mark.parametrize("build_fn", [sqlite_build, pg_build])
    def test_all_empty_returns_empty_string(self, build_fn) -> None:
        txt = build_fn(None, None, None, None)
        assert txt == ""


class TestHashConsistencyWithEmbedder:
    """Verify that the hash stored at upsert time matches populate_embeddings text."""

    @pytest.fixture()
    def db(self, tmp_path, monkeypatch):
        d = UnifiedWikiDB(tmp_path / "t.wiki.db", embedding_dim=8)
        d._vec_available = True
        monkeypatch.setattr(d, "upsert_embeddings_batch", lambda pairs: None)
        yield d
        d.close()

    def _seen_texts(self, db: UnifiedWikiDB) -> list[str]:
        """Run populate_embeddings and capture all texts sent to the embedder."""
        seen: list[str] = []

        def embedder(texts):
            seen.extend(texts)
            return [[0.0] * 8 for _ in texts]

        db.populate_embeddings(embedder, batch_size=64, architectural_only=False)
        return seen

    def test_node_with_llm_summary_and_symbol_name_hash_matches(self, db) -> None:
        """When a node has both llm_summary and symbol_name, the embedded text
        must equal _build_embedding_text(summary, None, None, name) — i.e.
        just the summary (symbol_name is not last-resort when summary exists).
        """
        db.upsert_node(
            "n1",
            rel_path="x.py",
            file_name="x.py",
            language="python",
            symbol_name="my_func",
            symbol_type="function",
            is_architectural=1,
            source_text="def my_func(): pass",
        )
        # Stamp an llm_summary directly
        db.update_node_summary("n1", "Handles async dispatch.", model_name="m")

        texts = self._seen_texts(db)
        assert texts, "embedder was never called"
        assert any("Handles async dispatch." in t for t in texts), (
            f"llm_summary not in embedded text; got: {texts}"
        )
        assert not any("my_func" in t and "Handles async dispatch." not in t for t in texts), (
            "symbol_name appeared as a standalone embed without llm_summary"
        )

    def test_second_run_is_noop_when_hash_matches(self, db) -> None:
        """After first embed, second run must be a no-op (hash match means skip)."""
        db.upsert_node(
            "n1",
            rel_path="x.py",
            file_name="x.py",
            language="python",
            symbol_name="my_func",
            symbol_type="function",
            is_architectural=1,
            source_text="def my_func(): pass",
        )
        db.update_node_summary("n1", "Summary text.", model_name="m")

        # First embed
        self._seen_texts(db)

        # Second embed should see nothing
        seen2 = self._seen_texts(db)
        assert seen2 == [], (
            "Second populate_embeddings must be a no-op when embedding text unchanged; "
            f"saw: {seen2}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fix 2 — SQLite IN-clause chunking (> 999 parameters)
# ═══════════════════════════════════════════════════════════════════════════


class TestSQLiteINClauseChunking:
    """upsert_nodes_batch must not crash with > 999 nodes."""

    @pytest.fixture()
    def db(self, tmp_path):
        d = UnifiedWikiDB(tmp_path / "big.wiki.db", embedding_dim=8)
        yield d
        d.close()

    def test_upsert_1001_nodes_does_not_crash(self, db) -> None:
        """Pre-fetching existing llm_summary for 1001 nodes previously crashed
        SQLite with 'too many SQL variables' (limit = 999 per statement).
        """
        nodes = [
            {
                "node_id": f"node_{i}",
                "rel_path": "x.py",
                "file_name": "x.py",
                "language": "python",
                "symbol_name": f"func_{i}",
                "symbol_type": "function",
                "source_text": f"def func_{i}(): pass",
            }
            for i in range(1001)
        ]
        # Should not raise sqlite3.OperationalError: too many SQL variables
        db.upsert_nodes_batch(nodes)
        count = db.conn.execute("SELECT COUNT(*) FROM repo_nodes").fetchone()[0]
        assert count == 1001

    def test_upsert_2000_nodes_does_not_crash(self, db) -> None:
        """Stress test with 2000 nodes (covers multiple chunks of 900)."""
        nodes = [
            {
                "node_id": f"node_{i}",
                "rel_path": f"f{i}.py",
                "file_name": f"f{i}.py",
                "language": "python",
                "symbol_name": f"func_{i}",
                "symbol_type": "function",
                "source_text": f"def func_{i}(): pass",
            }
            for i in range(2000)
        ]
        db.upsert_nodes_batch(nodes)
        count = db.conn.execute("SELECT COUNT(*) FROM repo_nodes").fetchone()[0]
        assert count == 2000


# ═══════════════════════════════════════════════════════════════════════════
# Fix 3 — _parse_numbered_list discards preamble
# ═══════════════════════════════════════════════════════════════════════════


class TestParseNumberedList:
    def test_plain_numbered_list_no_preamble(self) -> None:
        text = "1. First item\n2. Second item\n3. Third item"
        items = _parse_numbered_list(text, expected=3)
        assert items == ["First item", "Second item", "Third item"]

    def test_preamble_before_first_item_is_discarded(self) -> None:
        """The LLM sometimes adds a preface sentence before '1.'.

        Old code: re.split returns ["preamble\\n", "item1...", ...]
        and all non-empty parts were kept, so preamble counted as an item.
        New code: parts[0] is always discarded.
        """
        text = "Here are the summaries you requested:\n\n1. First item\n2. Second item"
        items = _parse_numbered_list(text, expected=2)
        assert len(items) == 2
        assert "Here are the summaries" not in items
        assert items[0] == "First item"
        assert items[1] == "Second item"

    def test_empty_preamble_still_works(self) -> None:
        """When there is no preamble, parts[0] is '' — discarding it is safe."""
        text = "1. Alpha\n2. Beta"
        items = _parse_numbered_list(text, expected=2)
        assert items == ["Alpha", "Beta"]

    def test_expected_count_respected(self) -> None:
        text = "Preamble text\n1. A\n2. B\n3. C\n4. D"
        items = _parse_numbered_list(text, expected=3)
        assert len(items) == 3
        assert items == ["A", "B", "C"]

    def test_preamble_causes_len_mismatch_without_fix(self) -> None:
        """Demonstrate the exact failure mode the fix addresses.

        Without the fix, a preamble inflates the item count → the batch
        parser sees len(parsed) == expected+1 → falls back to per-node.
        With the fix, preamble is stripped → len == expected → batch path used.
        """
        preamble = "Summary of 3 symbols:\n\n"
        text = preamble + "1. Item one\n2. Item two\n3. Item three"
        items = _parse_numbered_list(text, expected=3)
        assert len(items) == 3, (
            f"preamble caused len {len(items)} instead of 3: {items}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fix 4 — per-node fallback logs instead of swallowing errors
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchSummariseLogging:
    """Per-node fallback errors must be logged, not silently swallowed."""

    def test_per_node_failure_logged_as_warning(self, caplog) -> None:
        """When per-node LLM call raises, a WARNING must be emitted."""

        class _AlwaysFailLLM:
            def invoke(self, prompt: str):
                raise RuntimeError("simulated per-node LLM failure")

        with caplog.at_level(logging.WARNING, logger="app.core.filesystem_indexer"):
            results = _batch_summarise(_AlwaysFailLLM(), ["def foo(): pass"], batch_size=1)

        # Result should be None (node stays eligible for retry)
        assert results == [None]
        # Warning must have been emitted
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "No WARNING logged for per-node fallback failure"
        assert any(
            "per-node fallback failed" in r.message or "per-node fallback failed" in str(r.msg)
            for r in warnings
        ), f"Expected 'per-node fallback failed' in warning; got: {[r.msg for r in warnings]}"

    def test_per_node_failure_result_stays_none(self) -> None:
        """Failed per-node calls must leave the result as None (retry-eligible)."""

        class _AlwaysFailLLM:
            def invoke(self, _):
                raise ValueError("network error")

        results = _batch_summarise(_AlwaysFailLLM(), ["text1", "text2"], batch_size=5)
        assert results == [None, None], f"Expected [None, None], got {results}"
