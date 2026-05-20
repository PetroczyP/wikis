"""Unit tests for the Archive upload + extract service (#archive).

Exercises ``app.services.archive_upload_service.extract_archive`` directly
through synthetic in-memory archives. No HTTP/FastAPI dependency — keeps
the suite fast and pinpoints the security gates (traversal, symlink,
size, ratio, cleanup) at their actual implementation site.
"""

from __future__ import annotations

import io
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

from app.config import get_settings
from app.services.archive_upload_service import (
    ArchiveUploadError,
    extract_archive,
    resolve_archive_upload_id,
)


@pytest.fixture(autouse=True)
def _archive_root(tmp_path, monkeypatch):
    """Point ``archive_root`` at a per-test tmpdir.

    ``get_settings`` reads env at call time (no module-level cache to
    clear), so a ``monkeypatch.setenv`` here is enough.
    """
    root = tmp_path / "archives"
    monkeypatch.setenv("ARCHIVE_ROOT", str(root))
    # Cap defaults to 200 MB; tests use a small cap so oversize cases run fast.
    monkeypatch.setenv("ARCHIVE_MAX_SIZE_MB", "1")
    # Reload settings cache if the project introduces one later.
    get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None
    yield root


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    """Build an in-memory zip with the given (name, data) entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def _zip_with_symlink(link_name: str, target: str) -> bytes:
    """Build a zip carrying a single symlink entry."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo(link_name)
        # 0o120000 = S_IFLNK in the upper bits of external_attr.
        info.create_system = 3  # Unix
        info.external_attr = (0o120777 & 0xFFFF) << 16
        zf.writestr(info, target)
    return buf.getvalue()


def _tar_with_symlink() -> bytes:
    """Build a tar carrying a single symlink entry."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        info.mtime = int(time.time())
        tf.addfile(info)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_extract_zip_happy_path(_archive_root):
    """A simple zip extracts and the result fields match the on-disk state."""
    payload = _zip_bytes([
        ("README.md", b"hello"),
        ("src/main.py", b"print('hi')"),
    ])
    result = extract_archive("repo.zip", io.BytesIO(payload), declared_size=len(payload))
    assert result.file_count == 2
    assert result.size_bytes == len(b"hello") + len(b"print('hi')")
    # ``upload_id`` is the only token exposed over the wire; it must
    # resolve to the extraction directory server-side.
    extracted = resolve_archive_upload_id(result.upload_id)
    assert extracted.is_dir()
    assert (extracted / "README.md").read_text() == "hello"
    assert (extracted / "src" / "main.py").read_text() == "print('hi')"


def test_extract_targz_happy_path(_archive_root):
    """A simple .tar.gz extracts and produces the expected files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in [("a.txt", b"A"), ("dir/b.txt", b"BB")]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    payload = buf.getvalue()
    result = extract_archive("repo.tar.gz", io.BytesIO(payload), declared_size=len(payload))
    assert result.file_count == 2
    extracted = resolve_archive_upload_id(result.upload_id)
    assert (extracted / "a.txt").read_text() == "A"
    assert (extracted / "dir" / "b.txt").read_text() == "BB"


# ---------------------------------------------------------------------------
# Security: path traversal
# ---------------------------------------------------------------------------


def test_rejects_path_traversal_entry(_archive_root, tmp_path):
    """A zip entry named ``../etc/passwd`` must be refused."""
    payload = _zip_bytes([("../etc/passwd", b"r00t:x:0:0::/:/bin/sh")])
    with pytest.raises(ArchiveUploadError, match="escapes the extraction root"):
        extract_archive("evil.zip", io.BytesIO(payload), declared_size=len(payload))
    # No orphan extraction root should remain on disk.
    archives_root = Path(get_settings().archive_root).resolve()
    if archives_root.exists():
        assert list(archives_root.iterdir()) == []


def test_rejects_absolute_entry(_archive_root):
    """An absolute archive entry must be refused."""
    payload = _zip_bytes([("/etc/passwd", b"x")])
    with pytest.raises(ArchiveUploadError, match="absolute path"):
        extract_archive("evil.zip", io.BytesIO(payload), declared_size=len(payload))


# ---------------------------------------------------------------------------
# Security: symlinks
# ---------------------------------------------------------------------------


def test_rejects_zip_symlink(_archive_root):
    """A zip carrying a symlink entry must be refused."""
    payload = _zip_with_symlink("link", "/etc/passwd")
    with pytest.raises(ArchiveUploadError, match="symlink"):
        extract_archive("evil.zip", io.BytesIO(payload), declared_size=len(payload))


def test_rejects_tar_symlink(_archive_root):
    """A tar carrying a symlink entry must be refused."""
    payload = _tar_with_symlink()
    with pytest.raises(ArchiveUploadError, match="link entry"):
        extract_archive("evil.tar", io.BytesIO(payload), declared_size=len(payload))


# ---------------------------------------------------------------------------
# Security: size / extension / bomb
# ---------------------------------------------------------------------------


def test_rejects_oversize_body(_archive_root):
    """A body larger than ``archive_max_size_mb`` is refused with 413."""
    # Cap is 1 MB in the fixture. Build a 2 MB blob — extension check
    # happens first, so use a .zip filename.
    big = b"\x00" * (2 * 1024 * 1024)
    with pytest.raises(ArchiveUploadError) as exc:
        extract_archive("big.zip", io.BytesIO(big), declared_size=len(big))
    assert exc.value.status_code == 413


def test_rejects_bad_extension(_archive_root):
    """An unsupported extension (``.rar``) is rejected with 400."""
    with pytest.raises(ArchiveUploadError, match="Unsupported archive type"):
        extract_archive("evil.rar", io.BytesIO(b"\x00"), declared_size=1)


def test_rejects_zip_bomb_ratio(_archive_root):
    """A highly compressible archive exceeding 10× its on-disk size is refused."""
    # 11 MB of zeros compresses to a handful of KB → ratio ≈ thousands.
    # Cap is 1 MB so we keep the payload small but compressible enough
    # that the extracted bytes blow past 10× the compressed bytes.
    big_payload = b"\x00" * (11 * 1024 * 1024)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("zeros.bin", big_payload)
    payload = buf.getvalue()
    # Bump cap so we get past the body-size gate and hit the ratio gate.
    # Re-call extract_archive with a manually raised cap by adjusting env
    # via the existing fixture knob.
    import os
    os.environ["ARCHIVE_MAX_SIZE_MB"] = "50"
    try:
        with pytest.raises(ArchiveUploadError, match="zip bomb|compressed-size limit"):
            extract_archive(
                "bomb.zip", io.BytesIO(payload), declared_size=len(payload)
            )
    finally:
        os.environ["ARCHIVE_MAX_SIZE_MB"] = "1"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_malformed_archive_leaves_no_orphan(_archive_root):
    """A malformed archive cleans up after itself — no leftover extraction dir."""
    payload = b"this is not a real zip file"
    archives_root = Path(get_settings().archive_root).resolve()
    pre = set(archives_root.iterdir()) if archives_root.exists() else set()
    with pytest.raises(ArchiveUploadError):
        extract_archive("bad.zip", io.BytesIO(payload), declared_size=len(payload))
    post = set(archives_root.iterdir()) if archives_root.exists() else set()
    # The extraction root for this upload must be gone.
    assert post == pre


# ---------------------------------------------------------------------------
# resolve_archive_upload_id
# ---------------------------------------------------------------------------


def test_resolve_archive_upload_id_happy_path(_archive_root):
    """A freshly extracted upload resolves to its on-disk directory."""
    payload = _zip_bytes([("README.md", b"x")])
    result = extract_archive("ok.zip", io.BytesIO(payload), declared_size=len(payload))
    resolved = resolve_archive_upload_id(result.upload_id)
    assert resolved.is_dir()
    # Must be a direct child of the configured archive root.
    assert resolved.parent.resolve() == Path(get_settings().archive_root).resolve()


def test_resolve_archive_upload_id_missing_dir(_archive_root):
    """An unknown but well-formed upload_id raises 404."""
    fake = "0" * 32
    with pytest.raises(ArchiveUploadError) as exc:
        resolve_archive_upload_id(fake)
    assert exc.value.status_code == 404


def test_resolve_archive_upload_id_bad_format(_archive_root):
    """Malformed upload_ids are rejected with 400 before any disk access."""
    for bad in ("", "not-a-uuid", "ZZZZ" * 8, "a" * 31, "a" * 33):
        with pytest.raises(ArchiveUploadError) as exc:
            resolve_archive_upload_id(bad)
        assert exc.value.status_code == 400


def test_resolve_archive_upload_id_path_traversal(_archive_root):
    """Path-traversal payloads in the upload_id string are refused."""
    for evil in ("../foo", "..", "/etc/passwd", "../../etc", "abc/../def"):
        with pytest.raises(ArchiveUploadError) as exc:
            resolve_archive_upload_id(evil)
        # The regex catches these as malformed → 400.
        assert exc.value.status_code == 400


def test_rejects_dot_entry_zip(_archive_root):
    """A zip entry named '.' (all segments filter to empty) must be rejected.

    Previously ``_safe_destination`` would return the extraction root itself
    for such an entry; the downstream ``dest.open('wb')`` on a directory would
    raise ``IsADirectoryError`` (caught as OSError → cleanup), but the intent
    was never clear.  The explicit guard added by the Rio review makes the
    rejection immediate and unambiguous.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # Force a *file* entry named '.' (no trailing slash → not flagged is_dir).
        info = zipfile.ZipInfo(".")
        zf.writestr(info, b"should not land on disk")
    payload = buf.getvalue()
    with pytest.raises(ArchiveUploadError, match="resolves to the extraction root"):
        extract_archive("dot.zip", io.BytesIO(payload), declared_size=len(payload))
