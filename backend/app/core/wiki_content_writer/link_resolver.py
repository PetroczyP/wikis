"""Wiki-link resolver (#241, parent epic #227).

After the writer emits all pages, this module validates every ``[[…]]``
reference against the planned page index and rewrites the markdown in-place:

- **matched**   — target exactly matches a page title → kept as-is.
- **ambiguous** — no exact match but slug-distance finds ≥1 candidate →
                  link markup kept, target replaced with the best-match title.
- **missing**   — no candidate at all → ``[[…]]`` stripped, anchor text kept.

Consumed by the wiring step (#243).  Parsing relies on
``parsers.markdown_links.extract_links`` (#228) which already strips code
fences and inline code so embedded ``[[…]]`` in examples are silently ignored.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.parsers.markdown_links import Link, extract_links

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class LinkAction(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


@dataclass
class LinkResolution:
    """Resolution record for a single ``[[…]]`` reference."""

    target: str
    action: LinkAction
    # Canonical title when action is AMBIGUOUS; None otherwise.
    best_match: str | None = None
    # Slug-similarity score for AMBIGUOUS matches; 0.0 otherwise. Useful
    # for downstream logging when deciding whether a rewrite was confident.
    score: float = 0.0
    # For chaptered wikis: page path including chapter folder, e.g.
    # ``"authentication-service/rest-api"`` or ``"workers-chapter/_index"``.
    # ``None`` for non-chaptered wikis or unresolved links.
    resolved_path: str | None = None


@dataclass
class PageReport:
    """Per-page resolution results."""

    links: list[LinkResolution]
    rewritten: str


@dataclass
class ResolverReport:
    """Aggregate report across all pages."""

    pages: dict[str, PageReport] = field(default_factory=dict)

    @property
    def total_matched(self) -> int:
        return sum(1 for pr in self.pages.values() for lr in pr.links if lr.action == LinkAction.MATCHED)

    @property
    def total_ambiguous(self) -> int:
        return sum(1 for pr in self.pages.values() for lr in pr.links if lr.action == LinkAction.AMBIGUOUS)

    @property
    def total_missing(self) -> int:
        return sum(1 for pr in self.pages.values() for lr in pr.links if lr.action == LinkAction.MISSING)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Lower-case, strip accents, collapse non-alphanumeric runs to hyphens."""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _SLUG_RE.sub("-", normalized.lower()).strip("-")


def _slug_similarity(a: str, b: str) -> float:
    """Simple character-level overlap between two slugs (Jaccard on bigrams)."""
    sa, sb = _slugify(a), _slugify(b)
    if sa == sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    bigrams_a = {sa[i : i + 2] for i in range(len(sa) - 1)}
    bigrams_b = {sb[i : i + 2] for i in range(len(sb) - 1)}
    if not bigrams_a or not bigrams_b:
        # Single-character slugs: fall back to exact equality after slug
        return 1.0 if sa == sb else 0.0
    intersection = bigrams_a & bigrams_b
    union = bigrams_a | bigrams_b
    return len(intersection) / len(union)


def _find_best_match(target: str, index: list[str]) -> tuple[str | None, float]:
    """Return the best-matching page title and its score (0.0–1.0).

    Exact case-insensitive match wins immediately (score 1.0).
    Otherwise the highest slug-similarity score wins; ties broken by index order.
    Threshold: candidates with score < 0.3 are not returned.
    """
    target_lower = target.lower()
    best_title: str | None = None
    best_score = 0.0

    for title in index:
        # Exact case-insensitive match
        if title.lower() == target_lower:
            return title, 1.0
        score = _slug_similarity(target, title)
        if score > best_score:
            best_score = score
            best_title = title

    if best_score >= 0.3:
        return best_title, best_score
    return None, 0.0


# Matches [[…]] patterns (including pipe variants) for targeted replacement.
_WIKILINK_FULL_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Code-fence / inline-code regex — fenced + inline must stay in sync with
# markdown_links.py (#228). Kept local + position-preserving so the rewriter
# sees the same spans extract_links() used when deciding which [[…]] to
# surface. Indented-code masking is *not* mirrored here on purpose: the
# pairing loop can only rewrite wikilinks extract_links returns, so unmasked
# indented matches harmlessly fail to pair and are ignored.
#
# The `(?:```|$)` tail handles unclosed fences — common in LLM output where
# a doc cuts off mid-code-block. Without the alternation a `[[…]]` after the
# unclosed fence is incorrectly surfaced as a real wikilink.
_FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _mask_code_spans(text: str) -> str:
    """Blank out fenced and inline code spans, preserving offsets."""

    def _blank(m: re.Match[str]) -> str:
        return " " * (m.end() - m.start())

    text = _FENCED_BLOCK_RE.sub(_blank, text)
    text = _INLINE_CODE_RE.sub(_blank, text)
    return text


def _rebuild_inner(target: str, anchor: str | None, display: str | None) -> str:
    """Reassemble the inner text of a [[…]] from its parts."""
    body = target
    if anchor:
        body = f"{body}#{anchor}"
    if display is not None and display != target:
        body = f"{body}|{display}"
    return body


def _rewrite_page(
    body: str,
    links: list[Link],
    index_set: set[str],
    index: list[str],
    title_to_path: dict[str, str] | None = None,
) -> tuple[str, list[LinkResolution]]:
    """Rewrite *body* resolving each wikilink; return (rewritten_body, resolutions)."""
    if not links:
        return body, []

    # Mask code spans so the rewriter only sees real wikilinks — same view
    # extract_links() worked from. Offsets are preserved by the blank fill,
    # so splice ranges remain valid against the original body.
    masked = _mask_code_spans(body)
    wikilink_matches = list(_WIKILINK_FULL_RE.finditer(masked))

    # Filter to only the wikilink-kind links from extract_links output.
    wiki_links = [link for link in links if link.kind == "wikilink"]

    # Pair each Link with its corresponding regex match (same document order).
    pairs: list[tuple[re.Match[str], Link]] = []
    match_idx = 0
    for link in wiki_links:
        while match_idx < len(wikilink_matches):
            m = wikilink_matches[match_idx]
            inner = m.group(1).strip()
            raw_target = inner.split("|", 1)[0].strip()
            raw_target_clean = raw_target.split("#", 1)[0].strip()
            if raw_target_clean == link.target or raw_target == link.target:
                pairs.append((m, link))
                match_idx += 1
                break
            match_idx += 1

    # Build resolutions in document order alongside positions; apply splices
    # in reverse order so earlier offsets stay valid.
    positioned: list[tuple[int, LinkResolution]] = []
    splices: list[tuple[int, int, str]] = []  # (start, end, replacement)

    # Build reverse map: page_id_path → title, so path-based targets like
    # ``[[authentication-service/rest-api]]`` can be resolved to their title.
    path_to_title: dict[str, str] = {}
    if title_to_path:
        for t, p in title_to_path.items():
            path_to_title[p] = t

    for match, link in pairs:
        target = link.target
        anchor = link.anchor
        # Detect pipe-display syntax: when no pipe is present,
        # `Link.text` mirrors the inner text — which already contains the
        # anchor. Compare against the no-pipe reconstruction so an
        # anchor-only [[X#a]] isn't mistaken for [[X|display]].
        no_pipe_form = f"{target}#{anchor}" if anchor else target
        had_display = link.text != no_pipe_form
        display = link.text if had_display else None

        if target in index_set:
            # Exact title match — check if it maps to a chaptered path.
            resolved_path = title_to_path.get(target) if title_to_path else None
            positioned.append((
                match.start(),
                LinkResolution(target=target, action=LinkAction.MATCHED, resolved_path=resolved_path),
            ))
            continue

        # Check if the target is a chaptered path (e.g. ``chapter-slug/page-slug``).
        # This handles the case where the writer emits path-based wikilinks.
        if "/" in target and target in path_to_title:
            canonical_title = path_to_title[target]
            resolved_path = target  # the target IS the path
            positioned.append((
                match.start(),
                LinkResolution(
                    target=target,
                    action=LinkAction.AMBIGUOUS,
                    best_match=canonical_title,
                    score=1.0,
                    resolved_path=resolved_path,
                ),
            ))
            # Rewrite path-based link to canonical title for SPA compatibility.
            new_inner = _rebuild_inner(canonical_title, anchor, display)
            splices.append((match.start(), match.end(), f"[[{new_inner}]]"))
            continue

        best_title, score = _find_best_match(target, index)

        if best_title is not None:
            resolved_path = title_to_path.get(best_title) if title_to_path else None
            positioned.append(
                (
                    match.start(),
                    LinkResolution(
                        target=target,
                        action=LinkAction.AMBIGUOUS,
                        best_match=best_title,
                        score=score,
                        resolved_path=resolved_path,
                    ),
                )
            )
            # Preserve anchor / display while normalising target.
            new_inner = _rebuild_inner(best_title, anchor, display)
            splices.append((match.start(), match.end(), f"[[{new_inner}]]"))
        else:
            positioned.append((match.start(), LinkResolution(target=target, action=LinkAction.MISSING)))
            # Missing — strip [[…]] keeping readable text. Prefer the
            # pipe-display when present; otherwise the clean target so the
            # `#anchor` portion of `[[Page#section]]` doesn't leak into prose.
            replacement = display if display is not None else target
            splices.append((match.start(), match.end(), replacement))

    # Apply splices end-to-start so earlier offsets remain valid.
    result = body
    for start, end, replacement in sorted(splices, key=lambda s: s[0], reverse=True):
        result = result[:start] + replacement + result[end:]

    resolutions = [r for _, r in positioned]
    return result, resolutions


def resolve_wikilinks(
    pages: dict[str, str],
    page_index: list[str],
    chaptered_paths: dict[str, str] | None = None,
) -> ResolverReport:
    """Validate and rewrite ``[[wiki-links]]`` in all generated pages.

    Args:
        pages: ``{page_title: markdown_body}`` — the writer's emitted pages.
        page_index: Valid page titles from the planner's ``WikiStructureSpec``.
        chaptered_paths: Optional mapping of ``{page_title: page_id_path}``
            for chaptered wikis (#287).  When provided:

            - Each matched/ambiguous :class:`LinkResolution` gains a
              ``resolved_path`` attribute (e.g. ``"authentication-service/rest-api"``
              or ``"workers-chapter/_index"``) so callers can generate correct
              file-system or URL paths without re-deriving the slug.
            - Path-based wikilinks (``[[chapter-slug/page-slug]]``) are
              rewritten to the canonical page title so the SPA's title-based
              navigation continues to work.

    Returns:
        A :class:`ResolverReport` with per-page resolution lists and rewritten
        markdown.  Each link is classified as matched, ambiguous, or missing.
    """
    report = ResolverReport()

    if not pages:
        return report

    index_set: set[str] = set(page_index)

    for page_title, body in pages.items():
        links = extract_links(body)
        rewritten, resolutions = _rewrite_page(body, links, index_set, page_index, chaptered_paths)
        report.pages[page_title] = PageReport(links=resolutions, rewritten=rewritten)

    return report
