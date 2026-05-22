"""Page formatter for wiki content writer output (#320).

Two-pass formatter:
  Pass 1 (deterministic, no LLM):
    Replace inline ``[path:line]`` / ``[path:lo-hi]`` citation tokens with
    ``[^N]`` markdown footnote refs and append a ``## References`` block.

  Pass 2 (LLM, low-tier):
    Reorganise sections for human readability, subject to a content-richness
    guard that skips the LLM call for sparse or diagnostic content.

Design decisions
----------------
Two-pass separation
    Pass 1 is always applied first so footnote refs are always stable
    (deterministic, no I/O).  Pass 2 is an enhancement — failure is non-fatal
    and falls back to Pass 1 output.

Idempotency
    If ``## References`` is already present the formatter returns unchanged so
    repeated application does not double-annotate a page.

Richness guard
    The LLM Pass 2 is skipped when content is sparse (< 300 chars stripped,
    < 3 non-empty paragraphs) or when the LLM produces a known confusion
    marker.  This prevents low-value LLM calls on stub pages.

Public API
----------
``PageFormatter(llm_client)``
    *llm_client* — any LangChain ``BaseChatModel`` (or ``None`` to skip
    Pass 2 entirely).

``PageFormatter.format_page(markdown, page_spec=None) → str``
    Apply the two-pass formatter.  Never raises; on Pass 2 failure the
    Pass 1 output is returned.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.prompts import load_prompt

logger = logging.getLogger(__name__)

# ── Citation token patterns ───────────────────────────────────────────────────

# Matches multi-citation brackets: [a.py:1, b.py:2-3] (comma-separated)
# Captured as a single bracket with raw content for pre-splitting.
_MULTI_CITATION_RE = re.compile(
    r"\[([^:\[\]\n]+:\d+(?:-\d+)?(?:,\s*[^:\[\]\n]+:\d+(?:-\d+)?)+)\](?!\()"
)

# Matches a single citation element: path:N or path:lo-hi
_SINGLE_ELEMENT_RE = re.compile(r"([^:\[\]\n,]+):(\d+)(?:-(\d+))?")

# Matches [path:N] or [path:lo-hi] NOT immediately followed by '('
_CITATION_TOKEN_RE = re.compile(
    r"\[([^:\[\]\n]+):(\d+)(?:-(\d+))?\](?!\()"
)

# ── Frontmatter stripper ─────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

# ── Confusion markers emitted by an over-prompted formatter LLM ──────────────

_CONFUSION_MARKERS = (
    "I appreciate you sharing this context",
    "I cannot reorganize",
)

# ── LLM formatting prompt ─────────────────────────────────────────────────────

_FORMAT_PROMPT = load_prompt("wiki/page_format.md")


class PageFormatter:
    """Two-pass wiki page formatter.

    Parameters
    ----------
    llm_client :
        LangChain ``BaseChatModel`` used for Pass 2 section reorganisation.
        Pass ``None`` to run Pass 1 only (no LLM calls).
    """

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    # ── Public API ────────────────────────────────────────────────────────────

    def format_page(self, markdown: str, page_spec: Any = None) -> str:  # noqa: ARG002
        """Format *markdown* through two passes.

        Parameters
        ----------
        markdown :
            Raw markdown string produced by the writer/gate pipeline.
        page_spec :
            Optional ``PageSpec`` from the planner; reserved for future use
            (e.g. passing the page title into the LLM prompt).

        Returns
        -------
        str
            Formatted markdown.  Returns *markdown* unchanged if it is empty,
            already has ``## References``, or has no citation tokens and fails
            the richness guard.
        """
        if not markdown or not markdown.strip():
            return markdown

        # Pass 1 (deterministic) — convert [path:line] → [^N] footnotes
        pass1_output = self._pass1_footnotes(markdown)

        # Pass 2 (LLM reorganisation) disabled — agentic writers now produce
        # well-structured content directly. LLM reorganisation added noise
        # without improving readability at current content quality levels.
        return pass1_output

    # ── Pass 1: deterministic footnote rewriting ──────────────────────────────

    def _pass1_footnotes(self, markdown: str) -> str:
        """Replace citation tokens with footnote refs and append References block.

        Idempotent: if ``## References`` already exists, return *markdown*
        unchanged.  If no citation tokens are found, also return unchanged (no
        empty block appended).
        """
        if "## References" in markdown:
            return markdown

        # Find all citation tokens and assign stable footnote numbers.
        token_to_number: dict[str, int] = {}  # canonical token → footnote N
        footnote_defs: list[tuple[int, str]] = []  # (N, definition text)

        def _replace(m: re.Match[str]) -> str:
            path = m.group(1)
            lo = int(m.group(2))
            hi_str = m.group(3)
            hi = int(hi_str) if hi_str is not None else lo

            # Canonical token key (normalised)
            token_key = f"{path}:{lo}" if lo == hi else f"{path}:{lo}-{hi}"

            if token_key not in token_to_number:
                n = len(token_to_number) + 1
                token_to_number[token_key] = n
                if lo == hi:
                    defn = f"`{path}` line {lo}"
                else:
                    defn = f"`{path}` lines {lo}-{hi}"
                footnote_defs.append((n, defn))

            return f"[^{token_to_number[token_key]}]"

        def _expand_multi(m: re.Match[str]) -> str:
            """Expand [a.py:1, b.py:2-3] → [a.py:1][b.py:2-3] before main pass."""
            parts = [p.strip() for p in m.group(1).split(",")]
            return "".join(f"[{p}]" for p in parts)

        expanded = _MULTI_CITATION_RE.sub(_expand_multi, markdown)
        rewritten = _CITATION_TOKEN_RE.sub(_replace, expanded)

        if not token_to_number:
            # No citation tokens found — return original unchanged.
            return markdown

        # Build the ## References block.
        ref_lines = ["## References", ""]
        for n, defn in sorted(footnote_defs):
            ref_lines.append(f"[^{n}]: {defn}")
        ref_block = "\n".join(ref_lines)

        # Append with two blank lines separator.
        return rewritten.rstrip("\n") + "\n\n" + ref_block + "\n"

    # ── Pass 2: LLM section reorganisation ───────────────────────────────────

    @staticmethod
    def _passes_richness_guard(content: str) -> bool:
        """Return True if *content* is rich enough for LLM formatting.

        Guard conditions (any False → skip LLM):
        - At least 300 characters after stripping frontmatter.
        - At least 3 non-empty paragraphs (excluding the References block).

        Note: confusion-marker detection runs in ``_pass2_llm`` on the LLM
        output, not here — Pass 1 output is deterministic and cannot contain
        those markers.
        """
        stripped = _FRONTMATTER_RE.sub("", content)

        if len(stripped) < 300:
            return False

        # Count paragraphs excluding the ## References section
        body = stripped.split("## References")[0]
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if len(paragraphs) < 3:
            return False

        return True

    def _pass2_llm(self, pass1_output: str) -> str:
        """Send *pass1_output* to the LLM for section reorganisation.

        Falls back to *pass1_output* on:
        - Any exception from the LLM.
        - LLM output shorter than 50 % of *pass1_output*.
        - LLM output containing a known confusion marker.
        """
        try:
            prompt = _FORMAT_PROMPT.format(content=pass1_output)
            response = self._llm.invoke(prompt)
            llm_text: str = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("[formatter] Pass 2 LLM call failed: %s", exc)
            return pass1_output

        # Sanity checks on LLM output.
        if not llm_text or not llm_text.strip():
            logger.debug("[formatter] Pass 2 returned empty output; using Pass 1")
            return pass1_output

        for marker in _CONFUSION_MARKERS:
            if marker in llm_text:
                logger.debug("[formatter] Pass 2 confusion marker detected; using Pass 1")
                return pass1_output

        if len(llm_text) < len(pass1_output) * 0.5:
            logger.debug(
                "[formatter] Pass 2 output too short (%d vs %d); using Pass 1",
                len(llm_text),
                len(pass1_output),
            )
            return pass1_output

        return llm_text
