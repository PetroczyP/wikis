"""Tests for #322 incremental wiki page index — skip re-indexing unchanged pages.

Contract verified here:

* ``content_hash`` is deterministic: same markdown → same hash.
* Saving a page stores ``content_hash``; saving again with the same content
  does not clear ``embedding_content_hash``.
* ``populate_wiki_page_index`` skips pages where
  ``content_hash == embedding_content_hash``.
* After a content change (new ``content_hash``), the page is re-indexed and
  ``embedding_content_hash`` is updated.
* Pre-#322 rows (NULL ``embedding_content_hash``) are indexed once then stamped.
* An empty stale set is a no-op (``index_fn`` never called, returns 0).
"""

from __future__ import annotations

import pytest

from app.core.storage.incremental import compute_content_hash
from app.core.unified_db import UnifiedWikiDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "incr_page.wiki.db", embedding_dim=8)
    yield d
    d.close()


def _page(page_id: str, wiki_id: str = "w1", **overrides) -> dict:
    base = {
        "page_id": page_id,
        "wiki_id": wiki_id,
        "title": page_id,
        "anchor_slug": page_id,
        "content_hash": None,
    }
    base.update(overrides)
    return base


def _read_hashes(db: UnifiedWikiDB, wiki_id: str = "w1") -> dict[str, tuple]:
    """Return {page_id: (content_hash, embedding_content_hash)} for all pages."""
    rows = db.conn.execute(
        "SELECT page_id, content_hash, embedding_content_hash "
        "FROM wiki_pages WHERE wiki_id = ?",
        (wiki_id,),
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


class _RecordingIndexFn:
    """Records which page_ids were passed to the index callback."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, pages: list[dict]) -> None:
        self.calls.append([p["page_id"] for p in pages])

    @property
    def indexed_ids(self) -> list[str]:
        return [pid for batch in self.calls for pid in batch]


# ---------------------------------------------------------------------------
# Unit: _page_content_hash (compute_content_hash)
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self) -> None:
        assert compute_content_hash("hello") == compute_content_hash("hello")

    def test_different_content_different_hash(self) -> None:
        assert compute_content_hash("hello") != compute_content_hash("world")

    def test_empty_returns_empty_string(self) -> None:
        assert compute_content_hash("") == ""

    def test_whitespace_normalization(self) -> None:
        # Trailing spaces on a line should not change the hash.
        h1 = compute_content_hash("line one\nline two")
        h2 = compute_content_hash("line one   \nline two  ")
        assert h1 == h2


# ---------------------------------------------------------------------------
# Unit: content_hash stored in wiki_pages on upsert
# ---------------------------------------------------------------------------


class TestContentHashStoredOnUpsert:
    def test_content_hash_is_stored(self, db: UnifiedWikiDB) -> None:
        chash = compute_content_hash("# Hello\nSome content")
        db.upsert_wiki_page(_page("p1", content_hash=chash))
        row = db.get_wiki_page("p1")
        assert row is not None
        assert row["content_hash"] == chash

    def test_embedding_hash_defaults_to_null(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1", content_hash="abc123"))
        row = db.get_wiki_page("p1")
        assert row["embedding_content_hash"] is None

    def test_re_upsert_does_not_clear_embedding_hash(self, db: UnifiedWikiDB) -> None:
        """Re-saving a page must not clobber the embedding_content_hash stamp."""
        db.upsert_wiki_page(_page("p1", content_hash="hash-v1"))
        # Simulate a stamp written by populate_wiki_page_index.
        db.conn.execute(
            "UPDATE wiki_pages SET embedding_content_hash = 'hash-v1' "
            "WHERE page_id = 'p1'"
        )
        db.conn.commit()
        # Re-upsert the same page (e.g. after a partial regen).
        db.upsert_wiki_page(_page("p1", content_hash="hash-v1"))
        row = db.get_wiki_page("p1")
        # Stamp must survive.
        assert row["embedding_content_hash"] == "hash-v1"

    def test_re_upsert_with_changed_content_hash_updates_content_hash(
        self, db: UnifiedWikiDB,
    ) -> None:
        db.upsert_wiki_page(_page("p1", content_hash="hash-v1"))
        db.conn.execute(
            "UPDATE wiki_pages SET embedding_content_hash = 'hash-v1' "
            "WHERE page_id = 'p1'"
        )
        db.conn.commit()
        # Page content changed → new content_hash.
        db.upsert_wiki_page(_page("p1", content_hash="hash-v2"))
        row = db.get_wiki_page("p1")
        assert row["content_hash"] == "hash-v2"
        # Stamp remains what it was (stale) — populate_wiki_page_index picks it up.
        assert row["embedding_content_hash"] == "hash-v1"


# ---------------------------------------------------------------------------
# Unit: populate_wiki_page_index skip logic
# ---------------------------------------------------------------------------


class TestPopulateWikiPageIndex:
    def test_no_pages_returns_zero(self, db: UnifiedWikiDB) -> None:
        fn = _RecordingIndexFn()
        assert db.populate_wiki_page_index("w1", fn) == 0
        assert fn.calls == []

    def test_new_page_is_indexed_and_stamped(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1", content_hash="aaa"))
        fn = _RecordingIndexFn()

        n = db.populate_wiki_page_index("w1", fn)

        assert n == 1
        assert fn.indexed_ids == ["p1"]
        hashes = _read_hashes(db)
        assert hashes["p1"] == ("aaa", "aaa")  # embedding_hash now matches

    def test_unchanged_page_is_skipped(self, db: UnifiedWikiDB) -> None:
        """Second call with the same content_hash must be a no-op."""
        db.upsert_wiki_page(_page("p1", content_hash="aaa"))
        db.populate_wiki_page_index("w1", _RecordingIndexFn())

        fn2 = _RecordingIndexFn()
        n = db.populate_wiki_page_index("w1", fn2)

        assert n == 0
        assert fn2.calls == []

    def test_changed_page_is_reindexed(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1", content_hash="v1"))
        db.populate_wiki_page_index("w1", _RecordingIndexFn())

        # Simulate content change → new hash.
        db.upsert_wiki_page(_page("p1", content_hash="v2"))

        fn = _RecordingIndexFn()
        n = db.populate_wiki_page_index("w1", fn)

        assert n == 1
        assert fn.indexed_ids == ["p1"]
        hashes = _read_hashes(db)
        assert hashes["p1"] == ("v2", "v2")

    def test_unchanged_page_not_reindexed_while_changed_is(
        self, db: UnifiedWikiDB,
    ) -> None:
        db.upsert_wiki_page(_page("p1", anchor_slug="p1", content_hash="v1"))
        db.upsert_wiki_page(_page("p2", anchor_slug="p2", content_hash="v1"))
        db.populate_wiki_page_index("w1", _RecordingIndexFn())

        # Only p1 changes.
        db.upsert_wiki_page(_page("p1", anchor_slug="p1", content_hash="v2"))

        fn = _RecordingIndexFn()
        n = db.populate_wiki_page_index("w1", fn)

        assert n == 1
        assert fn.indexed_ids == ["p1"]

    def test_pre_322_null_embedding_hash_indexed_once_then_stamped(
        self, db: UnifiedWikiDB,
    ) -> None:
        """Existing rows with NULL embedding_content_hash are indexed once."""
        db.upsert_wiki_page(_page("p1", content_hash="abc"))
        # Ensure embedding_content_hash is NULL (it is by default, but be explicit).
        db.conn.execute(
            "UPDATE wiki_pages SET embedding_content_hash = NULL WHERE page_id = 'p1'"
        )
        db.conn.commit()

        fn = _RecordingIndexFn()
        n = db.populate_wiki_page_index("w1", fn)
        assert n == 1
        assert fn.indexed_ids == ["p1"]

        # Second call — now stamped, must be skipped.
        fn2 = _RecordingIndexFn()
        n2 = db.populate_wiki_page_index("w1", fn2)
        assert n2 == 0

    def test_null_content_hash_always_reindexed(self, db: UnifiedWikiDB) -> None:
        """Pages with NULL content_hash (pre-regen) are always re-indexed."""
        db.upsert_wiki_page(_page("p1", content_hash=None))

        fn = _RecordingIndexFn()
        n = db.populate_wiki_page_index("w1", fn)
        # Page is stale (content_hash IS NULL) → indexed.
        assert n == 1

    def test_scoped_to_wiki_id(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1", wiki_id="w1", anchor_slug="p1-w1", content_hash="aaa"))
        db.upsert_wiki_page(_page("p2", wiki_id="w2", anchor_slug="p2-w2", content_hash="bbb"))

        fn = _RecordingIndexFn()
        n = db.populate_wiki_page_index("w1", fn)

        assert n == 1
        assert fn.indexed_ids == ["p1"]

    def test_index_fn_receives_page_fields(self, db: UnifiedWikiDB) -> None:
        """index_fn must receive at minimum page_id, wiki_id, title, content_hash."""
        db.upsert_wiki_page(
            _page("p1", title="Overview", content_hash="deadbeef")
        )
        received: list[dict] = []

        def capturing_fn(pages: list[dict]) -> None:
            received.extend(pages)

        db.populate_wiki_page_index("w1", capturing_fn)

        assert len(received) == 1
        p = received[0]
        assert p["page_id"] == "p1"
        assert p["wiki_id"] == "w1"
        assert p["title"] == "Overview"
        assert p["content_hash"] == "deadbeef"


# ---------------------------------------------------------------------------
# Integration: index_wiki_pages — skip unchanged pages in WikiManagementService
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def _make_wiki_management():
    """Spin up an in-memory WikiManagementService for integration tests."""
    from unittest.mock import AsyncMock, MagicMock

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.db import _add_missing_columns, _ensure_wiki_page_fts
    from app.models.db_models import Base
    from app.services.wiki_management import WikiManagementService

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _add_missing_columns(engine)
    await _ensure_wiki_page_fts(engine)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    storage = MagicMock()  # ArtifactStorage — not used in index_wiki_pages
    svc = WikiManagementService(storage=storage, session_factory=session_factory)
    return svc, engine


class TestIndexWikiPagesSkip:
    """WikiManagementService.index_wiki_pages() must skip unchanged pages."""

    @pytest.fixture()
    async def svc(self):
        service, _ = await _make_wiki_management()
        return service

    @pytest.mark.asyncio
    async def test_first_run_indexes_all_pages(self, svc) -> None:
        pages = {"p1": "# Page 1\nContent", "p2": "# Page 2\nContent"}
        count = await svc.index_wiki_pages("wiki-1", pages)
        assert count == 2

    @pytest.mark.asyncio
    async def test_second_run_with_same_content_skips_all(self, svc) -> None:
        pages = {"p1": "# Page 1\nContent", "p2": "# Page 2\nContent"}
        await svc.index_wiki_pages("wiki-1", pages)
        # Second run — content unchanged.
        count = await svc.index_wiki_pages("wiki-1", pages)
        assert count == 0

    @pytest.mark.asyncio
    async def test_only_changed_page_is_reindexed(self, svc) -> None:
        pages = {"p1": "# Page 1\nContent A", "p2": "# Page 2\nContent B"}
        await svc.index_wiki_pages("wiki-1", pages)

        # Only p1 changes.
        pages_v2 = {"p1": "# Page 1\nContent CHANGED", "p2": "# Page 2\nContent B"}
        count = await svc.index_wiki_pages("wiki-1", pages_v2)
        assert count == 1

    @pytest.mark.asyncio
    async def test_removed_page_is_deleted(self, svc) -> None:
        from sqlalchemy import select

        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.db import _add_missing_columns, _ensure_wiki_page_fts
        from app.models.db_models import Base, WikiPageRecord
        from app.services.wiki_management import WikiManagementService
        from unittest.mock import MagicMock

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await _add_missing_columns(engine)
        await _ensure_wiki_page_fts(engine)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        svc2 = WikiManagementService(
            storage=MagicMock(), session_factory=session_factory,
        )

        await svc2.index_wiki_pages("wiki-1", {"p1": "# P1", "p2": "# P2"})
        # Remove p2.
        await svc2.index_wiki_pages("wiki-1", {"p1": "# P1"})

        async with session_factory() as session:
            rows = (await session.execute(
                select(WikiPageRecord).where(WikiPageRecord.wiki_id == "wiki-1")
            )).scalars().all()
        assert len(rows) == 1
