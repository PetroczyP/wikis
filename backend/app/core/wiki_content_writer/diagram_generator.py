"""Architecture diagram generator for wiki pages (#234).

Deterministic — no LLM calls.  Queries the storage graph and emits a
Mermaid ``graph LR`` block showing intra-cluster component relationships.

Public API
----------
``DiagramGenerator(storage)``
    ``generate_cluster_diagram(cluster_id, page_spec) -> str``
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_MAX_NODES = 15
# Relationship types to include in the diagram (structural + call graph).
_INTRA_CLUSTER_REL_TYPES = {
    "calls", "imports", "references", "creates",
    "member_uses", "inheritance", "composition",
}


def _safe_id(name: str) -> str:
    """Convert a symbol name to a Mermaid-safe node identifier."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _sanitize_label(name: str) -> str:
    """Sanitize a symbol name for use as a Mermaid double-quoted node label.

    Mermaid double-quoted labels break on `"`, `#`, unmatched `<`/`>`, and
    raw HTML entities.  Strip or replace these before embedding in the diagram.
    """
    # Replace double quotes (would close the label string early)
    s = name.replace('"', "'")
    # Remove characters that confuse Mermaid's label parser
    s = re.sub(r"[<>{}|\\]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s or _safe_id(name)


def _validate_mermaid(diagram: str) -> tuple[bool, str]:
    """Lightweight structural validation for a Mermaid diagram block.

    Returns (is_valid, error_reason).  Checks:
    - Fenced code block with mermaid tag
    - Valid graph/flowchart header
    - All node-declaration lines use quoted labels
    - No obvious unclosed quotes
    """
    lines = diagram.strip().splitlines()
    if not lines or lines[0].strip() != "```mermaid":
        return False, "missing ```mermaid fence"
    if lines[-1].strip() != "```":
        return False, "unclosed code fence"

    body = [l for l in lines[1:-1] if l.strip()]
    if not body:
        return False, "empty diagram body"

    first = body[0].strip()
    if not re.match(r"^(graph|flowchart)\s+(LR|TD|TB|RL|BT)$", first):
        return False, f"invalid header: {first!r}"

    for line in body[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Count unescaped double-quotes: should be even
        if stripped.count('"') % 2 != 0:
            return False, f"unmatched quote in: {stripped!r}"

    return True, ""


def _node_name(node: dict) -> str:
    """Extract display name from a node dict.

    The storage returns ``symbol_name`` (SQLite/Postgres real column); tests
    pass ``name`` (mocked dicts matching the spec).  Accept both.
    """
    return node.get("symbol_name") or node.get("name") or node["node_id"]


class DiagramGenerator:
    def __init__(self, storage: Any, llm_client: Any = None) -> None:
        self.storage = storage
        self._llm = llm_client

    def generate_cluster_diagram(self, cluster_id: int, page_spec: Any) -> str:
        """Return a ## Architecture section with Mermaid diagram + LLM context.

        The section includes:
        - A 2-3 sentence LLM-written explanation of the component relationships
          shown in the diagram (when an LLM client is provided).
        - The Mermaid graph block.

        Returns '' when no nodes are found for the cluster.
        """
        if self.storage is None:
            return ""

        try:
            mermaid = self._build_diagram(cluster_id)
        except Exception as exc:
            logger.debug("DiagramGenerator: cluster_id=%d error: %s", cluster_id, exc)
            return ""

        if not mermaid:
            return ""

        # Validate Mermaid syntax; ask LLM to fix if invalid.
        valid, reason = _validate_mermaid(mermaid)
        if not valid:
            logger.warning(
                "[DIAGRAM] cluster_id=%d invalid Mermaid (%s); attempting LLM fix",
                cluster_id, reason,
            )
            mermaid = self._fix_mermaid(mermaid, reason)
            if not mermaid:
                return ""

        # Build context: symbol names visible in the diagram
        try:
            nodes = self.storage.get_nodes_by_cluster(cluster_id)
            node_names = [_node_name(n) for n in _select_nodes(nodes)]
        except Exception:
            node_names = []

        explanation = self._explain(mermaid, node_names, page_spec)
        if explanation:
            return f"## Architecture\n\n{explanation}\n\n{mermaid}"
        return f"## Architecture\n\n{mermaid}"

    def _fix_mermaid(self, mermaid: str, reason: str) -> str:
        """Ask the LLM to fix invalid Mermaid syntax. Returns fixed diagram or ''."""
        if self._llm is None:
            return ""
        try:
            from langchain_core.messages import HumanMessage  # noqa: PLC0415
            prompt = (
                f"The following Mermaid diagram has a syntax error: {reason}\n\n"
                f"{mermaid}\n\n"
                f"Fix the Mermaid syntax so it renders correctly. "
                f"Return ONLY the corrected ```mermaid...``` block, nothing else. "
                f"Keep the same nodes and edges — only fix syntax issues."
            )
            response = self._llm.invoke([HumanMessage(content=prompt)])
            fixed = response.content if hasattr(response, "content") else str(response)
            # Extract the mermaid block from the response
            m = re.search(r"```mermaid\n.*?```", fixed, re.DOTALL)
            if m:
                candidate = m.group(0)
                ok, _ = _validate_mermaid(candidate)
                if ok:
                    return candidate
            logger.debug("[DIAGRAM] LLM fix did not produce valid Mermaid")
            return ""
        except Exception as exc:
            logger.debug("[DIAGRAM] LLM fix call failed: %s", exc)
            return ""

    def _explain(self, mermaid: str, node_names: list[str], page_spec: Any) -> str:
        """Ask the LLM for a 2-3 sentence explanation of the diagram."""
        if self._llm is None:
            return ""
        try:
            from langchain_core.messages import HumanMessage  # noqa: PLC0415
            title = getattr(page_spec, "title", "") or (
                page_spec.get("title", "") if isinstance(page_spec, dict) else ""
            )
            prompt = (
                f"The following Mermaid diagram shows the architecture of "
                f'"{title}". '
                f"Components: {', '.join(node_names[:12])}.\n\n"
                f"{mermaid}\n\n"
                f"Write 2-3 sentences describing the key relationships and "
                f"architectural pattern shown. Be specific and concise. "
                f"Do not repeat the component names as a list."
            )
            response = self._llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            return text.strip() if text and text.strip() else ""
        except Exception as exc:
            logger.debug("DiagramGenerator LLM explain failed: %s", exc)
            return ""

    # ── internals ──────────────────────────────────────────────────────────────

    def _build_diagram(self, cluster_id: int) -> str:
        nodes = self.storage.get_nodes_by_cluster(cluster_id)
        if not nodes:
            return ""

        nodes = _select_nodes(nodes)
        node_ids: set[str] = {n["node_id"] for n in nodes}
        id_to_name: dict[str, str] = {n["node_id"]: _node_name(n) for n in nodes}

        seen_edges: set[tuple[str, str]] = set()
        edges: list[tuple[str, str]] = []
        for node in nodes:
            nid = node["node_id"]
            try:
                raw_edges = self.storage.get_edges_from(
                    nid, rel_types=list(_INTRA_CLUSTER_REL_TYPES)
                )
            except Exception:
                raw_edges = []
            for e in raw_edges:
                target = e.get("target_id", "")
                if target in node_ids:
                    pair = (nid, target)
                    if pair not in seen_edges:
                        seen_edges.add(pair)
                        edges.append(pair)

        return _render(id_to_name, edges)


# ── helpers ────────────────────────────────────────────────────────────────────


def _select_nodes(nodes: list[dict]) -> list[dict]:
    """Filter to ≤ 15 nodes; prefer architectural ones when > 15 candidates."""
    if len(nodes) <= _MAX_NODES:
        return nodes
    arch = [n for n in nodes if n.get("is_architectural")]
    if len(arch) <= _MAX_NODES:
        return arch
    return arch[:_MAX_NODES]


def _render(id_to_name: dict[str, str], edges: list[tuple[str, str]]) -> str:
    """Build a Mermaid graph LR block from node names and edge pairs."""
    lines: list[str] = ["```mermaid", "graph LR"]

    # Emit node declarations — labels sanitized to prevent Mermaid parse errors.
    for nid, name in id_to_name.items():
        safe = _safe_id(name)
        label = _sanitize_label(name)
        lines.append(f'    {safe}["{label}"]')

    # Emit edges.
    for src_id, tgt_id in edges:
        src = _safe_id(id_to_name[src_id])
        tgt = _safe_id(id_to_name[tgt_id])
        lines.append(f"    {src} --> {tgt}")

    lines.append("```")
    return "\n".join(lines)
