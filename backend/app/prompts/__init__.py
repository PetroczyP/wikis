"""Prompt loader — reads Markdown prompt files from this package.

Usage
-----
    from app.prompts import load_prompt

    # Static prompt (no substitutions)
    text = load_prompt("writer/system.md")

    # Template prompt (Python .format()-style placeholders)
    text = load_prompt("planner/system.md").format(n_clusters=5)

Design
------
- All prompt files live under ``backend/app/prompts/`` as ``*.md`` files.
- ``load_prompt`` resolves paths relative to this file's directory.
- Results are cached via ``functools.lru_cache`` — no repeated I/O.
- Path separators use forward-slashes; works on all platforms.
"""

from __future__ import annotations

import functools
from pathlib import Path

_PROMPTS_ROOT: Path = Path(__file__).parent


@functools.lru_cache(maxsize=256)
def load_prompt(relative_path: str) -> str:
    """Return the text of a prompt Markdown file.

    Parameters
    ----------
    relative_path:
        Path relative to ``backend/app/prompts/``, e.g.
        ``"writer/system.md"`` or ``"planner/user_template.md"``.
        Use forward slashes regardless of OS.

    Returns
    -------
    str
        File contents, stripped of leading/trailing whitespace.

    Raises
    ------
    FileNotFoundError
        When the file does not exist at the resolved path.
    """
    path = (_PROMPTS_ROOT / relative_path).resolve()
    # Guard against path traversal (e.g. "../../etc/passwd").
    if not path.is_relative_to(_PROMPTS_ROOT.resolve()):
        raise ValueError(
            f"Prompt path escapes prompts directory: {relative_path!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt file not found: {path} "
            f"(resolved from relative_path={relative_path!r})"
        )
    return path.read_text(encoding="utf-8").strip()
