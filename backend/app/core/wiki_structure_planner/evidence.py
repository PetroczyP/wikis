"""Evidence pack builder for code clusters (#232).

Deterministic, parallel-friendly prefetch of ~1500 tokens of real code evidence
per cluster, so the planner LLM names clusters from actual signatures and file
heads rather than 5-line summaries.

Public API
----------
build_pack(cluster, kind="code", *, repo_root, top_k, token_budget) -> EvidencePack

EvidencePack
    .signatures   : list of (name, source_path, layer, summary)
    .file_heads   : list of (rel_path, text) — first 40 lines of entry/public files
    .readme_excerpt: str — up to 60 lines from first README in cluster dirs
    .sql_blocks   : list of str — CREATE TABLE blocks from *.sql files
    .serialize()  : str — render all sections for prompt injection (format TBD by #236)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._evidence_utils import safe_join
from .structure_skeleton import ArtifactInfo, Cluster

logger = logging.getLogger(__name__)

# ── Layer priority — lower number = higher priority ───────────────────────────
_LAYER_PRIORITY: dict[str, int] = {
    "entry_point": 0,
    "public_api": 1,
    "core_type": 2,
    "infrastructure": 3,
    "internal": 4,
    "constant": 5,
}
_DEFAULT_PRIORITY = 6  # anything unknown goes last

# Layers whose source files we read the head of (first N lines).
_FILE_HEAD_LAYERS: frozenset[str] = frozenset({"entry_point", "public_api"})

_README_NAMES: frozenset[str] = frozenset({"readme.md", "readme.rst", "readme.txt", "readme"})

_SQL_CREATE_PATTERN = re.compile(
    r"(CREATE\s+TABLE\b[^;]*;)",
    re.IGNORECASE | re.DOTALL,
)

# Approximate chars per token — conservative (LLaMA-family tokenisers are closer to 3.5).
_CHARS_PER_TOKEN = 4


# ── EvidencePack ──────────────────────────────────────────────────────────────


@dataclass
class EvidencePack:
    """Prefetched evidence for one cluster, ready for prompt injection.

    Fields are deliberately plain Python types so the planner agent (#236) can
    format them however it likes — the serialization here is a sensible default
    only; callers should override it when they control the prompt template.

    Attributes
    ----------
    cluster_id : int
    kind : str
        Source kind of the originating cluster ("code", "doc", …).
    signatures : list of (name, source_path, layer, summary)
        Top-K symbol signatures sorted by layer priority then connections desc.
    file_heads : list of (rel_path, text)
        First ≤40 lines of each unique entry_point / public_api source file.
    readme_excerpt : str
        Head ≤60 lines of the first README found inside the cluster's dirs.
    sql_blocks : list of str
        All ``CREATE TABLE …;`` blocks extracted from *.sql files in cluster dirs.
    artifact_names : list of str
        Architectural symbol names in the cluster (used by ``slice_evidence_pack``).
    path_leaves : list of str
        Repo-relative file paths for the cluster's artifacts.
    call_graph_edges : list of (caller, callee) tuples
        Internal call-graph edges among cluster symbols.
    cross_cluster_refs : list of str
        Symbol names from other clusters referenced by this cluster.
    doc_snippets : list of str
        Documentation file paths or excerpts attached to this cluster.
    """

    cluster_id: int
    kind: str
    signatures: list[tuple[str, str, str, str]] = field(default_factory=list)
    file_heads: list[tuple[str, str]] = field(default_factory=list)
    readme_excerpt: str = ""
    sql_blocks: list[str] = field(default_factory=list)
    # Fields used by slice_evidence_pack (#289) — populated by build_pack
    # and by callers that build packs for chaptered clusters.
    artifact_names: list[str] = field(default_factory=list)
    path_leaves: list[str] = field(default_factory=list)
    call_graph_edges: list[tuple] = field(default_factory=list)
    cross_cluster_refs: list[str] = field(default_factory=list)
    doc_snippets: list[str] = field(default_factory=list)

    def serialize(self) -> str:
        """Render all sections as a prompt-ready string.

        The format is intentionally simple — the planner agent (#236) wraps
        this inside its own prompt template; we just produce readable text.
        """
        parts: list[str] = [f"[EvidencePack cluster={self.cluster_id} kind={self.kind}]"]

        if self.signatures:
            parts.append("## Symbol Signatures")
            for name, src, layer, summary in self.signatures:
                line = f"  {layer:16s} {name} ({src})"
                if summary:
                    line += f" — {summary}"
                parts.append(line)

        if self.file_heads:
            parts.append("## File Heads")
            for rel_path, text in self.file_heads:
                parts.append(f"### {rel_path}")
                parts.append(text)

        if self.readme_excerpt:
            parts.append("## README")
            parts.append(self.readme_excerpt)

        if self.sql_blocks:
            parts.append("## Schema (SQL)")
            for block in self.sql_blocks:
                parts.append(block)

        return "\n".join(parts)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _layer_rank(layer: str) -> int:
    return _LAYER_PRIORITY.get(layer, _DEFAULT_PRIORITY)


def _sorted_artifacts(artifacts: list[ArtifactInfo]) -> list[ArtifactInfo]:
    """Sort by (layer priority asc, connections desc, name asc) for stable ordering."""
    return sorted(
        artifacts,
        key=lambda a: (_layer_rank(a.layer), -a.connections, a.name),
    )


def _read_file_head(repo_root: str, rel_path: str, max_lines: int = 40) -> str | None:
    """Read up to *max_lines* lines from *rel_path* under *repo_root*.

    Returns None when the path is unsafe or the file cannot be read.
    Uses streaming reads so a large source file is not fully loaded just
    to keep the first few dozen lines.
    """
    import itertools

    safe = safe_join(repo_root, rel_path)
    if safe is None:
        logger.debug("evidence: path escape rejected: %r", rel_path)
        return None
    try:
        with open(safe, encoding="utf-8", errors="replace") as fh:
            lines = list(itertools.islice(fh, max_lines))
    except (OSError, PermissionError):
        return None
    return "".join(lines).rstrip("\n")


def _find_readme(repo_root: str, dir_path: str) -> str | None:
    """Return content of the first README found in *dir_path* under *repo_root*, or None."""
    safe_dir = safe_join(repo_root, dir_path)
    if safe_dir is None:
        return None
    try:
        entries = os.listdir(safe_dir)
    except (OSError, PermissionError):
        return None
    for entry in sorted(entries):
        if entry.lower() in _README_NAMES:
            readme_rel = os.path.join(dir_path, entry)
            content = _read_file_head(repo_root, readme_rel, max_lines=60)
            if content is not None:
                return content
    return None


def _extract_sql_blocks(repo_root: str, dir_path: str) -> list[str]:
    """Return all CREATE TABLE blocks from *.sql files directly under *dir_path*."""
    safe_dir = safe_join(repo_root, dir_path)
    if safe_dir is None:
        return []
    blocks: list[str] = []
    try:
        entries = os.listdir(safe_dir)
    except (OSError, PermissionError):
        return []
    for entry in sorted(entries):
        if not entry.lower().endswith(".sql"):
            continue
        sql_rel = os.path.join(dir_path, entry)
        safe_sql = safe_join(repo_root, sql_rel)
        if safe_sql is None:
            continue
        try:
            text = Path(safe_sql).read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue
        for match in _SQL_CREATE_PATTERN.finditer(text):
            blocks.append(match.group(1).strip())
    return blocks


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from char count. Used for total-pack budgeting."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _truncate_signatures_to_budget(
    signatures: list[tuple[str, str, str, str]],
    budget_chars: int,
) -> list[tuple[str, str, str, str]]:
    """Drop low-priority signatures until the serialized block fits in *budget_chars*.

    Signatures are already sorted by priority (entry_point first), so we trim
    from the tail (lowest priority).
    """
    total = sum(len(f"  {s[2]:16s} {s[0]} ({s[1]}) — {s[3]}") + 1 for s in signatures)
    result = list(signatures)
    while total > budget_chars and len(result) > 1:
        dropped = result.pop()  # remove lowest-priority entry
        total -= len(f"  {dropped[2]:16s} {dropped[0]} ({dropped[1]}) — {dropped[3]}") + 1
    return result


# Per-section character caps (rough fractions of the total budget).
# When the total is 1500 tokens × 4 chars/token = 6000 chars, these split:
#   signatures 40%, file_heads 35%, README 15%, SQL 10%.
_SECTION_FRACTIONS = {
    "signatures": 0.40,
    "file_heads": 0.35,
    "readme": 0.15,
    "sql": 0.10,
}


def _enforce_total_budget(pack: EvidencePack, char_budget: int) -> None:
    """Cap each section then trim from the lowest priority if total still over.

    Order of truncation when over-budget: SQL → README → file_heads → signatures.
    Signatures are most informative for naming, so they're shed last.
    """
    file_head_cap = int(char_budget * _SECTION_FRACTIONS["file_heads"])
    readme_cap = int(char_budget * _SECTION_FRACTIONS["readme"])
    sql_cap = int(char_budget * _SECTION_FRACTIONS["sql"])

    pack.file_heads = _cap_file_heads(pack.file_heads, file_head_cap)
    pack.readme_excerpt = pack.readme_excerpt[:readme_cap]
    pack.sql_blocks = _cap_sql_blocks(pack.sql_blocks, sql_cap)

    while len(pack.serialize()) > char_budget:
        if pack.sql_blocks:
            pack.sql_blocks.pop()
        elif pack.readme_excerpt:
            pack.readme_excerpt = ""
        elif pack.file_heads:
            pack.file_heads.pop()
        elif len(pack.signatures) > 1:
            pack.signatures.pop()
        else:
            break  # one signature minimum, accept overflow


def _cap_file_heads(file_heads: list[tuple[str, str]], char_cap: int) -> list[tuple[str, str]]:
    """Keep file heads until adding the next one would exceed char_cap."""
    capped: list[tuple[str, str]] = []
    total = 0
    for path, head in file_heads:
        entry_len = len(path) + len(head) + 6  # rough header overhead
        if total + entry_len > char_cap and capped:
            break
        capped.append((path, head))
        total += entry_len
    return capped


def _cap_sql_blocks(blocks: list[str], char_cap: int) -> list[str]:
    """Keep SQL blocks while remaining under char_cap."""
    capped: list[str] = []
    total = 0
    for block in blocks:
        if total + len(block) > char_cap and capped:
            break
        capped.append(block)
        total += len(block)
    return capped


# ── Public API ────────────────────────────────────────────────────────────────


def build_pack(
    cluster: Cluster,
    kind: str = "code",
    *,
    repo_root: str,
    top_k: int = 20,
    token_budget: int = 1500,
) -> EvidencePack:
    """Build an evidence pack for *cluster*.

    Parameters
    ----------
    cluster : Cluster
        Code cluster whose ``artifacts`` (``ArtifactInfo(kind="symbol", …)``)
        are used as the evidence source.  Pass ``cluster.kind`` via the *kind*
        parameter if you need to override (rare).
    kind : str
        Defaults to ``"code"``.  Passed through to ``EvidencePack.kind``.
    repo_root : str
        Absolute path to the checked-out repository root.  All file IO is
        confined to this directory tree.
    top_k : int
        Maximum number of symbol signatures to include.
    token_budget : int
        Approximate total token budget for the serialized pack.  Truncation
        by layer priority is applied when the budget is exceeded.

    Returns
    -------
    EvidencePack
    """
    char_budget = token_budget * _CHARS_PER_TOKEN

    # ── 1. Signatures (sorted by layer priority, capped at top_k) ────────
    # Code clusters can hold heterogeneous artifacts; filter to symbols so
    # doc / jira / confluence entries don't leak into the signatures block.
    symbol_artifacts = [a for a in cluster.artifacts if a.kind == "symbol"]
    sorted_arts = _sorted_artifacts(symbol_artifacts)
    top_artifacts = sorted_arts[:top_k]
    signatures: list[tuple[str, str, str, str]] = [(a.name, a.source_path, a.layer, a.summary) for a in top_artifacts]

    # ── 2. File heads (entry_point / public_api layers only, deduplicated) ─
    seen_paths: set[str] = set()
    file_heads: list[tuple[str, str]] = []
    for art in top_artifacts:
        if art.layer not in _FILE_HEAD_LAYERS:
            continue
        if art.source_path in seen_paths:
            continue
        seen_paths.add(art.source_path)
        head = _read_file_head(repo_root, art.source_path)
        if head is not None:
            file_heads.append((art.source_path, head))

    # ── 3. README excerpt (first hit across cluster dirs) ────────────────
    readme_excerpt = ""
    for dir_path in cluster.dirs:
        content = _find_readme(repo_root, dir_path)
        if content is not None:
            readme_excerpt = content
            break

    # ── 4. SQL CREATE TABLE blocks ────────────────────────────────────────
    sql_blocks: list[str] = []
    for dir_path in cluster.dirs:
        sql_blocks.extend(_extract_sql_blocks(repo_root, dir_path))

    # ── 5. Per-section caps + total-budget enforcement ────────────────────
    # Signatures are most informative for naming, so they're shed last.
    sig_budget_chars = int(char_budget * _SECTION_FRACTIONS["signatures"])
    signatures = _truncate_signatures_to_budget(signatures, sig_budget_chars)

    # ── 6. Populate slice-helper fields ──────────────────────────────────
    # artifact_names: names of all symbol artifacts (not just top-k)
    artifact_names: list[str] = [a.name for a in symbol_artifacts]
    # path_leaves: unique source paths across all symbol artifacts
    path_leaves: list[str] = list(dict.fromkeys(a.source_path for a in symbol_artifacts))

    pack = EvidencePack(
        cluster_id=cluster.cluster_id,
        kind=kind,
        signatures=signatures,
        file_heads=file_heads,
        readme_excerpt=readme_excerpt,
        sql_blocks=sql_blocks,
        artifact_names=artifact_names,
        path_leaves=path_leaves,
        # call_graph_edges and cross_cluster_refs are not available from the
        # Cluster dataclass alone; callers that build richer packs should set
        # these directly after calling build_pack().
        call_graph_edges=[],
        cross_cluster_refs=[],
        doc_snippets=[],
    )
    _enforce_total_budget(pack, char_budget)
    return pack


# ── EvidenceSlice ─────────────────────────────────────────────────────────────


@dataclass
class EvidenceSlice:
    """Focused slice of a cluster's evidence pack for one sub-page (#289).

    Attributes
    ----------
    subpage_index : int
        Copied from ``SubPageSpec.page_order`` — the raw LLM integer, which
        may be 1-based or arbitrary. Do NOT use as an array offset.
    subpage_title : str
        Tentative title copied from the corresponding SubPageSpec.
    symbols : list of str
        Architectural symbol names for this subtopic.
    file_paths : list of str
        Repo-relative file paths relevant to the slice symbols.
    doc_paths : list of str
        Documentation file paths shared from the parent cluster pack.
    call_graph_edges : list of (caller, callee) tuples
        Call-graph edges where at least one endpoint is in ``symbols``.
    cross_cluster_refs : list of str
        Symbol names from other clusters whose names overlap with ``symbols``.
    summary : str
        Concatenated LLM summaries for the slice symbols (≤ 1 200 chars),
        or an empty string when no storage is available.
    """

    subpage_index: int
    subpage_title: str
    symbols: list[str]
    file_paths: list[str]
    doc_paths: list[str]
    call_graph_edges: list[tuple]
    cross_cluster_refs: list[str]
    summary: str


# ── slice_evidence_pack ───────────────────────────────────────────────────────

_SUMMARY_CHAR_CAP = 1200


def slice_evidence_pack(
    pack: "EvidencePack",
    chapter_spec: Any,
    *,
    storage: Any = None,
) -> "list[EvidenceSlice]":
    """Distribute a cluster's EvidencePack across the SubPageSpec entries.

    For each ``SubPageSpec`` in ``chapter_spec.subpages``, builds an
    ``EvidenceSlice`` by intersecting the pack's artifact names with
    the subpage's ``target_symbols``.  When ``target_symbols`` is empty,
    falls back to an even round-robin split.

    Parameters
    ----------
    pack : EvidencePack
        Full evidence pack for the cluster.
    chapter_spec : ChapterSpec
        Chapter produced by ``run_planner_chaptered`` — its ``subpages``
        list drives the slice boundaries.
    storage : optional
        Storage object with a ``get_node_summary(symbol: str) -> str``
        method.  When provided, the ``summary`` field of each slice is
        populated by concatenating the LLM summaries of its symbols up to
        ``_SUMMARY_CHAR_CAP`` characters.  Exceptions from ``storage`` are
        caught and result in an empty summary.

    Returns
    -------
    list[EvidenceSlice]
        One ``EvidenceSlice`` per ``SubPageSpec`` in the chapter.
    """
    subpages = chapter_spec.subpages
    n = len(subpages)
    all_symbols = pack.artifact_names

    slices: list[EvidenceSlice] = []

    for idx, sp in enumerate(subpages):
        # ── Symbols for this subpage ──────────────────────────────────────
        target = sp.target_symbols or []
        slice_symbols = [s for s in target if s in all_symbols]

        if not slice_symbols:
            # Even round-robin split using enumeration index, NOT page_order.
            # page_order is a free-form LLM integer that can be 1-based or
            # arbitrary; using it as an array index produces empty slices
            # (e.g. page_order=1, n=1 → start=len, end=2*len → empty).
            start = idx * len(all_symbols) // n
            end = (idx + 1) * len(all_symbols) // n
            slice_symbols = all_symbols[start:end]
            if not slice_symbols and all_symbols:
                # Degenerate case: rounding left this slice empty → give it all
                slice_symbols = all_symbols

        # ── File paths ───────────────────────────────────────────────────
        if slice_symbols:
            slice_paths = [
                p for p in pack.path_leaves
                if any(sym.lower() in p.lower() for sym in slice_symbols)
            ]
        else:
            slice_paths = []
        if not slice_paths:
            slice_paths = list(pack.path_leaves)

        # ── Call-graph edges ─────────────────────────────────────────────
        sym_set = set(slice_symbols)
        slice_edges = [
            (c, e) for (c, e) in pack.call_graph_edges
            if c in sym_set or e in sym_set
        ]

        # ── Cross-cluster refs ───────────────────────────────────────────
        slice_cross = [
            ref for ref in pack.cross_cluster_refs
            if any(sym.lower() in ref.lower() for sym in slice_symbols)
        ]

        # ── Shared doc context (up to 3 snippets from the pack) ──────────
        slice_docs = pack.doc_snippets[:3]

        # ── Summary from storage ─────────────────────────────────────────
        summary = ""
        if storage is not None:
            parts: list[str] = []
            for sym in slice_symbols:
                try:
                    text = storage.get_node_summary(sym)
                except Exception:  # noqa: BLE001
                    text = ""
                if text:
                    parts.append(text)
            if parts:
                raw = " ".join(parts)
                summary = raw[:_SUMMARY_CHAR_CAP]

        slices.append(EvidenceSlice(
            subpage_index=sp.page_order,
            subpage_title=sp.title,
            symbols=slice_symbols,
            file_paths=slice_paths,
            doc_paths=slice_docs,
            call_graph_edges=slice_edges,
            cross_cluster_refs=slice_cross,
            summary=summary,
        ))

    return slices
