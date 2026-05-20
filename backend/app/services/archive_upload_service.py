"""Archive upload + safe extraction for the wizard's "Archive" connector.

Accepts a single multipart ``.zip`` / ``.tar.gz`` / ``.tgz`` / ``.tar``
upload, extracts it under :func:`get_settings().archive_root` /
``<uuid>/``, and returns an opaque ``upload_id`` (UUID) the wizard
hands back on subsequent scan + generate calls. The backend resolves
the upload_id to ``<archive_root>/<upload_id>`` server-side via
:func:`resolve_archive_upload_id`; absolute host paths never cross
the wire.

Security gates applied at extraction time:

* **Path traversal** — every entry's resolved path must stay inside the
  per-upload extraction root. ``..`` segments, absolute entry names, and
  symlinks all fail closed.
* **Symlinks** — both ``zipfile`` and ``tarfile`` can carry symlink
  entries.  We refuse them outright; even if their target string looks
  innocent on inspection, a later read through the link could escape
  the extraction root or point at a host file the process can see.
* **Body cap** — request size is bounded by
  ``settings.archive_max_size_mb`` (default 200 MB).
* **Zip-bomb cap** — total extracted size may not exceed 10× the
  archive's on-disk size.
* **Cleanup** — any failure during extraction tears down the partially
  populated extraction root before raising so we don't leak orphan
  directories on disk.
"""

from __future__ import annotations

import logging
import re
import shutil
import tarfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from app.config import get_settings

logger = logging.getLogger(__name__)


# Permitted extensions. ``magic``-byte sniffing would be stronger but
# adds a dependency; extension-check + safe-extract is the MVP per spec.
_ALLOWED_EXTS: tuple[str, ...] = (".zip", ".tar.gz", ".tgz", ".tar")

# Post-extraction size must not exceed this multiple of the on-disk
# archive size. A typical text-heavy zip compresses ~5×; 10× is generous
# for normal content and decisive against a crafted zip bomb.
_MAX_EXPAND_RATIO: int = 10

# ``uuid.uuid4().hex`` produces 32 lowercase hex chars. We accept the same
# shape on the wire so a malformed upload_id is rejected before any disk
# I/O (Path traversal payloads like ``../foo`` fail this guard).
_UPLOAD_ID_RE = re.compile(r"^[a-f0-9]{32}$")


class ArchiveUploadError(Exception):
    """Raised when an archive upload is rejected or extraction fails.

    Carries an HTTP status code so the route can surface a precise
    response (400 for validation, 413 for oversize).
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ArchiveUploadResult:
    """Outcome of a successful upload + extraction."""

    upload_id: str
    extracted_path: str  # absolute path to the extraction root
    file_count: int
    size_bytes: int  # total bytes written under ``extracted_path``


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _archive_root() -> Path:
    """Resolve the configured archive root to an absolute Path.

    Resolving once at use time (rather than caching at import) lets tests
    point ``archive_root`` at a tmpdir via env vars without re-importing.
    """
    root = Path(get_settings().archive_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_archive_upload_id(upload_id: str) -> Path:
    """Resolve an ``upload_id`` to its extraction directory under ``archive_root``.

    The ``upload_id`` is the opaque token returned by
    ``POST /api/v1/uploads/archive`` — the only archive identifier that
    crosses the wire. The backend resolves it to ``<archive_root>/<uuid>``
    here so absolute host paths never leak to the frontend.

    Raises ``ArchiveUploadError(400)`` for malformed ids (rejects path
    traversal payloads like ``../foo`` before any disk access), and
    ``ArchiveUploadError(404)`` when the directory is missing or no longer
    a child of the configured archive root.
    """
    if not upload_id or not _UPLOAD_ID_RE.match(upload_id):
        raise ArchiveUploadError(
            f"Invalid archive upload_id: {upload_id!r}", status_code=400
        )
    root = _archive_root()
    target = (root / upload_id).resolve()
    # Direct child only — the same containment guard the validator used.
    # Rejects symlink/traversal even if the upload_id slipped the regex.
    if root not in target.parents:
        raise ArchiveUploadError(
            f"Archive upload_id not found: {upload_id}", status_code=404
        )
    if not target.is_dir():
        raise ArchiveUploadError(
            f"Archive upload_id not found: {upload_id}", status_code=404
        )
    return target


def extract_archive(
    filename: str | None,
    stream: IO[bytes],
    declared_size: int | None = None,
) -> ArchiveUploadResult:
    """Validate, save, and safely extract an uploaded archive.

    Parameters
    ----------
    filename:
        Original client filename. Used for extension validation; the
        on-disk name is randomised so a malicious basename can't collide
        with anything the operator cares about.
    stream:
        Synchronous file-like (the FastAPI ``UploadFile.file`` handle).
    declared_size:
        Optional client-declared size used for an early reject before we
        spool the body to disk. The real cap is re-checked after the
        upload lands (clients can lie).

    Returns
    -------
    ArchiveUploadResult
        ``upload_id`` is the opaque token returned to the wizard;
        ``extracted_path`` is an internal field used by tests and is
        not surfaced over the wire — resolved server-side via
        :func:`resolve_archive_upload_id`.

    Raises
    ------
    ArchiveUploadError
        On extension mismatch, oversize body, traversal/symlink entries,
        zip-bomb-shaped archives, or malformed input. The route maps the
        ``status_code`` straight onto the HTTP response.
    """
    settings = get_settings()
    max_bytes = settings.archive_max_size_mb * 1024 * 1024
    if declared_size is not None and declared_size > max_bytes:
        raise ArchiveUploadError(
            f"Archive exceeds the {settings.archive_max_size_mb} MB limit",
            status_code=413,
        )

    ext = _validated_extension(filename)
    upload_id = uuid.uuid4().hex
    root = _archive_root() / upload_id
    root.mkdir(parents=True, exist_ok=False)

    # Spool the upload to disk first — extracting from the in-memory
    # stream would require trusting Content-Length, and the zip/tar
    # readers want a seekable file anyway. The temp file lives next to
    # the extraction root so a single rmtree on failure cleans both up.
    archive_path = root / f"upload{ext}"
    try:
        archive_size = _spool_to_disk(stream, archive_path, max_bytes)
        max_extracted = archive_size * _MAX_EXPAND_RATIO
        file_count, total_size = _extract(archive_path, root, ext, max_extracted)
    except ArchiveUploadError:
        shutil.rmtree(root, ignore_errors=True)
        raise
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise ArchiveUploadError(f"Failed to read archive: {exc}") from exc
    finally:
        # The uploaded archive itself is not needed after extraction —
        # remove it whether extraction succeeded or failed (the rmtree
        # above already covers the failure case).
        try:
            archive_path.unlink(missing_ok=True)
        except OSError:
            pass

    logger.info(
        "Archive %s extracted: files=%d size=%d path=%s",
        upload_id, file_count, total_size, root,
    )
    return ArchiveUploadResult(
        upload_id=upload_id,
        extracted_path=str(root),
        file_count=file_count,
        size_bytes=total_size,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validated_extension(filename: str | None) -> str:
    """Return the canonical archive extension or raise.

    ``.tar.gz`` is a compound extension; check it before ``.gz`` so the
    inner ``.tar`` is preserved for tarfile's mode dispatch.
    """
    if not filename:
        raise ArchiveUploadError("Archive filename is required")
    name = filename.lower()
    if name.endswith(".tar.gz"):
        return ".tar.gz"
    for ext in (".tgz", ".zip", ".tar"):
        if name.endswith(ext):
            return ext
    raise ArchiveUploadError(
        "Unsupported archive type. Allowed: .zip, .tar.gz, .tgz, .tar"
    )


def _spool_to_disk(stream: IO[bytes], dest: Path, max_bytes: int) -> int:
    """Stream the upload to ``dest`` and enforce ``max_bytes``.

    Returns the on-disk size. Raises ``ArchiveUploadError(413)`` the
    moment the running total exceeds the cap, which short-circuits a
    rogue uploader before the rest of the body lands.
    """
    written = 0
    chunk_size = 1024 * 1024
    with dest.open("wb") as fh:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise ArchiveUploadError(
                    f"Archive exceeds the {max_bytes // (1024 * 1024)} MB limit",
                    status_code=413,
                )
            fh.write(chunk)
    return written


def _extract(
    archive_path: Path, root: Path, ext: str, max_extracted: int
) -> tuple[int, int]:
    """Dispatch to zip/tar extractor based on extension."""
    if ext == ".zip":
        return _extract_zip(archive_path, root, max_extracted)
    return _extract_tar(archive_path, root, ext, max_extracted)


def _safe_destination(root: Path, member_name: str) -> Path:
    """Resolve ``member_name`` under ``root`` and verify containment.

    Returns the resolved destination path. Raises
    ``ArchiveUploadError`` for absolute paths, ``..`` traversal, or
    anything that lands outside ``root`` after resolution.

    We pre-clean the relative name (strip leading separators, reject
    ``..`` segments) BEFORE the resolve so a symlinked extraction-root
    parent can't be exploited; the post-resolve containment check is
    belt-and-braces.
    """
    if not member_name:
        raise ArchiveUploadError("Archive contains an entry with an empty name")
    # Normalise to forward slashes so the same checks work on Windows
    # archives unpacked on POSIX hosts.
    cleaned = member_name.replace("\\", "/")
    if cleaned.startswith("/"):
        raise ArchiveUploadError(f"Archive entry has absolute path: {member_name}")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ArchiveUploadError(
            f"Archive entry escapes the extraction root: {member_name}"
        )
    if not parts:
        # All segments were empty or '.', e.g. name="." or "./" — nothing to
        # write under the extraction root.  Path(*[]) evaluates to Path('.')
        # which resolves to root itself; we reject explicitly rather than
        # relying on the downstream IsADirectoryError from dest.open('wb').
        raise ArchiveUploadError(
            f"Archive entry resolves to the extraction root: {member_name}"
        )
    dest = (root / Path(*parts)).resolve()
    root_resolved = root.resolve()
    if dest != root_resolved and root_resolved not in dest.parents:
        raise ArchiveUploadError(
            f"Archive entry escapes the extraction root: {member_name}"
        )
    return dest


def _extract_zip(archive_path: Path, root: Path, max_extracted: int) -> tuple[int, int]:
    """Extract a zip with per-entry traversal + size + symlink checks."""
    file_count = 0
    total = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            # Symlink detection: zip stores the unix mode in the upper
            # 16 bits of ``external_attr``. ``S_IFLNK`` (0o120000) marks
            # a symbolic link entry.
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and (unix_mode & 0o170000) == 0o120000:
                raise ArchiveUploadError(
                    f"Archive contains symlink (not allowed): {info.filename}"
                )
            dest = _safe_destination(root, info.filename)
            if info.is_dir() or info.filename.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, dest.open("wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_extracted:
                        raise ArchiveUploadError(
                            "Archive expands beyond the 10× compressed-size limit "
                            "(possible zip bomb)"
                        )
                    out.write(chunk)
            file_count += 1
    return file_count, total


def _extract_tar(
    archive_path: Path, root: Path, ext: str, max_extracted: int
) -> tuple[int, int]:
    """Extract a tar with per-entry traversal + size + symlink checks."""
    mode = "r:gz" if ext in (".tar.gz", ".tgz") else "r:"
    file_count = 0
    total = 0
    with tarfile.open(archive_path, mode) as tf:
        for member in tf:
            if member.issym() or member.islnk():
                raise ArchiveUploadError(
                    f"Archive contains link entry (not allowed): {member.name}"
                )
            if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
                raise ArchiveUploadError(
                    f"Archive contains a device/special entry: {member.name}"
                )
            dest = _safe_destination(root, member.name)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                # Unknown member type — be conservative and reject.
                raise ArchiveUploadError(
                    f"Archive contains unsupported entry type: {member.name}"
                )
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with extracted, dest.open("wb") as out:
                while True:
                    chunk = extracted.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_extracted:
                        raise ArchiveUploadError(
                            "Archive expands beyond the 10× compressed-size limit "
                            "(possible tar bomb)"
                        )
                    out.write(chunk)
            file_count += 1
    return file_count, total


__all__ = [
    "ArchiveUploadError",
    "ArchiveUploadResult",
    "extract_archive",
    "resolve_archive_upload_id",
]
