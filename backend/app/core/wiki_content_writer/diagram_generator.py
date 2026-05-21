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

    # Emit node declarations (label in quotes handles spaces/specials).
    for nid, name in id_to_name.items():
        safe = _safe_id(name)
        lines.append(f'    {safe}["{name}"]')

    # Emit edges.
    for src_id, tgt_id in edges:
        src = _safe_id(id_to_name[src_id])
        tgt = _safe_id(id_to_name[tgt_id])
        lines.append(f"    {src} --> {tgt}")

    lines.append("```")
    return "\n".join(lines)
