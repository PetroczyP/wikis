"""Unit tests for DiagramGenerator.generate_repo_diagram (Change 2).

Covers:
- generate_repo_diagram returns a valid Mermaid graph LR block
- Cluster nodes use slugified titles (≤30 chars)
- Cross-cluster edges (A→B) produce arrows in the diagram
- No intra-cluster edges appear (only cross-cluster)
- Empty / None storage returns ''
- Max 10 cluster nodes are shown
- Cluster nodes with no cross-cluster edges still appear as isolated nodes
- Returns '' on any storage error (graceful fallback)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.wiki_content_writer.diagram_generator import DiagramGenerator


# ── Helpers ────────────────────────────────────────────────────────────────────


def _node(
    node_id: str,
    name: str,
    macro_cluster: int,
    kind: str = "class",
    is_architectural: int = 1,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "name": name,
        "symbol_name": name,
        "kind": kind,
        "macro_cluster": macro_cluster,
        "is_architectural": is_architectural,
    }


def _edge(source_id: str, target_id: str, rel_type: str = "calls") -> dict[str, Any]:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "rel_type": rel_type,
        "weight": 1.0,
    }


def _make_storage_for_repo_diagram(
    nodes_by_cluster: dict[int, list[dict]],
    edges_map: dict[str, list[dict]] | None = None,
) -> MagicMock:
    """Build a mock storage for generate_repo_diagram.

    ``nodes_by_cluster`` maps cluster_id → list of node dicts.
    ``edges_map`` maps node_id → list of edge dicts.
    """
    storage = MagicMock()

    def _get_nodes_by_cluster(macro: int, micro: int | None = None, limit: int | None = 1000):
        return nodes_by_cluster.get(macro, [])

    storage.get_nodes_by_cluster.side_effect = _get_nodes_by_cluster

    if edges_map is not None:
        def _get_edges_from(node_id, rel_types=None):
            return edges_map.get(node_id, [])
        storage.get_edges_from.side_effect = _get_edges_from
    else:
        storage.get_edges_from.return_value = []

    return storage


# ── Scenario 1: Basic cross-cluster dependency ─────────────────────────────────


class TestBasicCrossClusterDiagram:
    """Two clusters with a cross-cluster edge → diagram shows both nodes + 1 arrow."""

    def _setup(self):
        # Cluster 1: node n1 (Auth cluster)
        # Cluster 2: node n2 (DB cluster)
        # Edge: n1 → n2 (cross-cluster)
        nodes = {
            1: [_node("n1", "AuthManager", macro_cluster=1)],
            2: [_node("n2", "Database", macro_cluster=2)],
        }
        edges_map = {
            "n1": [_edge("n1", "n2", "calls")],   # cross-cluster
            "n2": [],
        }
        storage = _make_storage_for_repo_diagram(nodes, edges_map)
        dg = DiagramGenerator(storage)
        cluster_titles = {1: "Auth System", 2: "Database Layer"}
        return dg, cluster_titles

    def test_diagram_is_non_empty(self):
        dg, cluster_titles = self._setup()
        result = dg.generate_repo_diagram(list(cluster_titles.keys()), cluster_titles)
        assert result != ""

    def test_diagram_has_mermaid_fence(self):
        dg, cluster_titles = self._setup()
        result = dg.generate_repo_diagram(list(cluster_titles.keys()), cluster_titles)
        assert "```mermaid" in result
        assert result.strip().endswith("```")

    def test_diagram_has_graph_lr(self):
        dg, cluster_titles = self._setup()
        result = dg.generate_repo_diagram(list(cluster_titles.keys()), cluster_titles)
        assert "graph LR" in result

    def test_diagram_contains_both_cluster_titles(self):
        dg, cluster_titles = self._setup()
        result = dg.generate_repo_diagram(list(cluster_titles.keys()), cluster_titles)
        # At least part of each cluster title should appear
        assert "Auth" in result
        assert "Database" in result

    def test_diagram_contains_cross_cluster_arrow(self):
        dg, cluster_titles = self._setup()
        result = dg.generate_repo_diagram(list(cluster_titles.keys()), cluster_titles)
        assert "-->" in result

    def test_diagram_has_exactly_one_arrow(self):
        dg, cluster_titles = self._setup()
        result = dg.generate_repo_diagram(list(cluster_titles.keys()), cluster_titles)
        assert result.count("-->") == 1


# ── Scenario 2: No cross-cluster edges ────────────────────────────────────────


class TestNoCrossClusterEdges:
    """Clusters with no cross-cluster edges → all nodes shown, no arrows."""

    def test_no_arrows_when_no_cross_cluster_edges(self):
        nodes = {
            1: [_node("n1", "ServiceA", macro_cluster=1)],
            2: [_node("n2", "ServiceB", macro_cluster=2)],
        }
        # All edges are intra-cluster — neither node_id crosses
        edges_map: dict[str, list] = {"n1": [], "n2": []}
        storage = _make_storage_for_repo_diagram(nodes, edges_map)
        dg = DiagramGenerator(storage)
        cluster_titles = {1: "Service A", 2: "Service B"}
        result = dg.generate_repo_diagram([1, 2], cluster_titles)

        # Diagram exists but has no arrows
        assert "```mermaid" in result
        assert "-->" not in result

    def test_isolated_clusters_still_appear_as_nodes(self):
        nodes = {
            1: [_node("n1", "ServiceA", macro_cluster=1)],
            2: [_node("n2", "ServiceB", macro_cluster=2)],
        }
        edges_map: dict[str, list] = {"n1": [], "n2": []}
        storage = _make_storage_for_repo_diagram(nodes, edges_map)
        dg = DiagramGenerator(storage)
        cluster_titles = {1: "Service Alpha", 2: "Service Beta"}
        result = dg.generate_repo_diagram([1, 2], cluster_titles)

        assert "Service" in result


# ── Scenario 3: Empty / None storage ─────────────────────────────────────────


class TestEmptyStorage:
    def test_none_storage_returns_empty_string(self):
        dg = DiagramGenerator(None)
        result = dg.generate_repo_diagram([1, 2], {1: "A", 2: "B"})
        assert result == ""

    def test_empty_cluster_ids_returns_empty_string(self):
        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = []
        dg = DiagramGenerator(storage)
        result = dg.generate_repo_diagram([], {})
        assert result == ""

    def test_storage_error_returns_empty_string(self):
        storage = MagicMock()
        storage.get_nodes_by_cluster.side_effect = RuntimeError("DB error")
        dg = DiagramGenerator(storage)
        result = dg.generate_repo_diagram([1], {1: "Cluster A"})
        assert result == ""


# ── Scenario 4: Max 10 cluster nodes ──────────────────────────────────────────


class TestMaxClusterNodes:
    """With more than 10 clusters, only 10 appear in the diagram."""

    def test_max_10_clusters_shown(self):
        # 15 clusters, each with 1 node
        nodes = {
            i: [_node(f"n{i}", f"Symbol{i}", macro_cluster=i)]
            for i in range(1, 16)
        }
        edges_map: dict[str, list] = {f"n{i}": [] for i in range(1, 16)}
        storage = _make_storage_for_repo_diagram(nodes, edges_map)
        dg = DiagramGenerator(storage)
        cluster_titles = {i: f"Cluster {i}" for i in range(1, 16)}
        result = dg.generate_repo_diagram(list(range(1, 16)), cluster_titles)

        # Count how many "Cluster N" labels appear
        present = sum(1 for i in range(1, 16) if f"Cluster {i}" in result or f"Cluster_{i}" in result)
        assert present <= 10

    def test_exactly_10_clusters_all_shown(self):
        nodes = {
            i: [_node(f"n{i}", f"Symbol{i}", macro_cluster=i)]
            for i in range(1, 11)
        }
        edges_map: dict[str, list] = {f"n{i}": [] for i in range(1, 11)}
        storage = _make_storage_for_repo_diagram(nodes, edges_map)
        dg = DiagramGenerator(storage)
        cluster_titles = {i: f"Cluster {i}" for i in range(1, 11)}
        result = dg.generate_repo_diagram(list(range(1, 11)), cluster_titles)

        present = sum(1 for i in range(1, 11) if f"Cluster {i}" in result)
        assert present == 10


# ── Scenario 5: Node label truncation ────────────────────────────────────────


class TestNodeLabelTruncation:
    """Cluster node labels are at most 30 chars."""

    def test_long_cluster_title_is_truncated(self):
        long_title = "A" * 40  # 40 chars — should be truncated to 30
        nodes = {1: [_node("n1", "Symbol1", macro_cluster=1)]}
        edges_map: dict[str, list] = {"n1": []}
        storage = _make_storage_for_repo_diagram(nodes, edges_map)
        dg = DiagramGenerator(storage)
        cluster_titles = {1: long_title}
        result = dg.generate_repo_diagram([1], cluster_titles)

        # The raw 40-char title must not appear verbatim
        assert long_title not in result
        # A truncated version (≤30 chars) should appear
        assert long_title[:30] in result

    def test_short_title_appears_unchanged(self):
        short_title = "Auth Core"
        nodes = {1: [_node("n1", "Symbol1", macro_cluster=1)]}
        edges_map: dict[str, list] = {"n1": []}
        storage = _make_storage_for_repo_diagram(nodes, edges_map)
        dg = DiagramGenerator(storage)
        cluster_titles = {1: short_title}
        result = dg.generate_repo_diagram([1], cluster_titles)

        assert short_title in result


# ── Scenario 6: Intra-cluster edges excluded ──────────────────────────────────


class TestIntraClusterEdgesExcluded:
    """Edges between nodes in the same cluster must NOT appear in the repo diagram."""

    def test_intra_cluster_edge_not_shown(self):
        # Both n1 and n2 are in cluster 1; edge between them is intra-cluster
        nodes = {
            1: [
                _node("n1", "Alpha", macro_cluster=1),
                _node("n2", "Beta", macro_cluster=1),
            ],
            2: [_node("n3", "Gamma", macro_cluster=2)],
        }
        edges_map = {
            "n1": [_edge("n1", "n2", "calls")],   # intra-cluster
            "n2": [],
            "n3": [],
        }
        storage = _make_storage_for_repo_diagram(nodes, edges_map)
        dg = DiagramGenerator(storage)
        cluster_titles = {1: "Cluster One", 2: "Cluster Two"}
        result = dg.generate_repo_diagram([1, 2], cluster_titles)

        # No arrows — the only edge is intra-cluster
        assert "-->" not in result


# ── Scenario 7: Valid Mermaid output ──────────────────────────────────────────


class TestValidMermaidOutput:
    """Output must be structurally valid Mermaid."""

    def _make_two_cluster_storage(self):
        nodes = {
            1: [_node("n1", "ServiceA", macro_cluster=1)],
            2: [_node("n2", "ServiceB", macro_cluster=2)],
        }
        edges_map = {
            "n1": [_edge("n1", "n2", "imports")],
            "n2": [],
        }
        return _make_storage_for_repo_diagram(nodes, edges_map)

    def test_diagram_starts_with_mermaid_fence(self):
        storage = self._make_two_cluster_storage()
        dg = DiagramGenerator(storage)
        result = dg.generate_repo_diagram([1, 2], {1: "Service A", 2: "Service B"})
        lines = result.strip().splitlines()
        assert lines[0].strip() == "```mermaid"

    def test_diagram_ends_with_closing_fence(self):
        storage = self._make_two_cluster_storage()
        dg = DiagramGenerator(storage)
        result = dg.generate_repo_diagram([1, 2], {1: "Service A", 2: "Service B"})
        assert result.strip().endswith("```")

    def test_no_unmatched_quotes(self):
        storage = self._make_two_cluster_storage()
        dg = DiagramGenerator(storage)
        result = dg.generate_repo_diagram([1, 2], {1: "Service A", 2: "Service B"})
        for line in result.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("```") and not stripped.startswith("graph"):
                # Each label line should have an even number of double-quotes
                assert stripped.count('"') % 2 == 0, f"Unmatched quote in: {stripped!r}"
