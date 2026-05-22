"""LLM prompts for the surgical-edit regime (#116 PR 3).

The surgical patcher is invoked when a page's only changes are
``MODIFIED`` (or ``MODIFIED + MOVED``) — i.e. the page structure is
intact, only specific symbols' source text shifted. The prompt's job is
to revise prose discussing those symbols without rewriting unrelated
sections, retitling the page, or altering citation paths that weren't
moved.

The quality gate in :class:`PagePatcher` compares the LLM's output to
the original via a token-set similarity score. If too much changed, the
patcher falls back to structural regen.
"""

from __future__ import annotations

from app.prompts import load_prompt

SURGICAL_EDIT_SYSTEM: str = load_prompt("wiki/surgical_edit_system.md")
SURGICAL_EDIT_USER_TEMPLATE: str = load_prompt("wiki/surgical_edit_user.md")


#: Per-source cap for the surgical-edit prompt. A typical changed
#: symbol is well under this; the cap protects against pathological
#: cases like a 1000-line class whose body would blow the LLM context
#: window. Picked at 2000 chars (~500 tokens) as a generous default;
#: tunable per-deployment via the constant or a future settings field.
MAX_SOURCE_CHARS_IN_PROMPT = 2000


def _truncate_and_fence_safe(text: str, *, limit: int = MAX_SOURCE_CHARS_IN_PROMPT) -> str:
    """Make a snippet safe to embed inside a ```fenced``` block.

    Two protections:
    * Truncate to ``limit`` chars so big symbols don't blow the context
      window. Appends an explicit truncation marker so the LLM knows
      it's seeing a partial view.
    * Replace triple-backticks with a visually similar Unicode sequence
      so the rendered prompt's fences stay well-formed.
    """
    if not text:
        return ""
    safe = text.replace("```", "ʼʼʼ")  # U+02BC modifier letter apostrophe
    if len(safe) > limit:
        return safe[:limit].rstrip() + "\n[...truncated for prompt budget...]"
    return safe.rstrip()


def format_symbol_diff(
    symbol_name: str,
    old_source: str,
    new_source: str,
    signature_change: str | None = None,
) -> str:
    """Render one symbol's before/after for inclusion in the user prompt.

    Kept as a standalone function so the prompt assembly is testable
    without instantiating an LLM client. Output is one block per
    symbol, separated by a blank line.

    Both source snippets pass through :func:`_truncate_and_fence_safe`
    so the prompt stays bounded and the fenced blocks can't be broken
    by triple-backticks lurking in the source itself.
    """
    parts = [f"### {symbol_name}"]
    if signature_change:
        parts.append(f"Signature change: {signature_change}")
    parts.append("```before")
    parts.append(_truncate_and_fence_safe(old_source))
    parts.append("```")
    parts.append("```after")
    parts.append(_truncate_and_fence_safe(new_source))
    parts.append("```")
    return "\n".join(parts)
