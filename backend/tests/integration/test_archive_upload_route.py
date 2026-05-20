"""Integration tests for ``POST /api/v1/uploads/archive`` (#283).

End-to-end exercise of the archive upload route through FastAPI's
TestClient. Validates the wire-schema (``upload_id`` shape) and the
two main rejection paths (oversize body → 413, bad extension → 400).

Uses the same ``client`` async fixture as the other integration tests
(in-memory SQLite, ``AUTH_ENABLED=false``).
"""

from __future__ import annotations

import io
import zipfile

import pytest


@pytest.fixture(autouse=True)
def _archive_settings(tmp_path, monkeypatch):
    """Point ``archive_root`` at a per-test tmpdir and cap size at 1 MB."""
    monkeypatch.setenv("ARCHIVE_ROOT", str(tmp_path / "archives"))
    monkeypatch.setenv("ARCHIVE_MAX_SIZE_MB", "1")
    from app.config import get_settings

    if hasattr(get_settings, "cache_clear"):
        get_settings.cache_clear()
    yield


def _build_zip(entries: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_upload_archive_happy_path(client):
    """A real zip uploads, extracts, and returns the upload_id shape."""
    payload = _build_zip([
        ("README.md", b"hello world"),
        ("src/main.py", b"print('hi')"),
    ])
    files = {"archive": ("repo.zip", payload, "application/zip")}
    resp = await client.post("/api/v1/uploads/archive", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # New wire shape: upload_id + counts only — no extracted_path.
    assert set(body.keys()) == {"upload_id", "file_count", "size_bytes"}
    assert isinstance(body["upload_id"], str) and len(body["upload_id"]) == 32
    assert body["file_count"] == 2
    assert body["size_bytes"] == len(b"hello world") + len(b"print('hi')")
    assert "extracted_path" not in body


@pytest.mark.asyncio
async def test_upload_archive_oversize_returns_413(client):
    """A body larger than ``ARCHIVE_MAX_SIZE_MB`` is refused with 413."""
    # Cap is 1 MB; send 2 MB of zero bytes with a .zip name so the
    # extension check passes and the size gate fires.
    big = b"\x00" * (2 * 1024 * 1024)
    files = {"archive": ("big.zip", big, "application/zip")}
    resp = await client.post("/api/v1/uploads/archive", files=files)
    assert resp.status_code == 413, resp.text


@pytest.mark.asyncio
async def test_upload_archive_bad_extension_returns_400(client):
    """An unsupported extension (.rar) is rejected with 400."""
    files = {"archive": ("evil.rar", b"\x00\x00", "application/octet-stream")}
    resp = await client.post("/api/v1/uploads/archive", files=files)
    assert resp.status_code == 400, resp.text
    assert "Unsupported archive type" in resp.text
