"""Unit tests for DiagramGenerator (#234).

Red-green-refactor: tests written first, run against the not-yet-existing
implementation to confirm they fail for the right reasons.

Three scenarios per spec:
- 3 nodes, 2 edges → diagram contains all node names + 2 arrows
- 0 nodes → empty string returned
- 20+ nodes with mix of architectural / non-architectural → only architectural
  ones included (up to 15 max)
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.wiki_content_writer.diagram_generator import DiagramGenerator


# ── Fake storage helpers ───────────────────────────────────────────────────────


def _node(
    node_id: str,
    name: str,
    kind: str = "class",
    file_path: str = "src/mod.py",
    is_architectural: int = 1,
    layer: str = "public_api",
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "name": name,
        "kind": kind,
        "file_path": file_path,
        "start_line": 1,
        "layer": layer,
        "is_architectural": is_architectural,
    }


def _edge(source_id: str, target_id: str, rel_type: str = "calls") -> dict[str, Any]:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "rel_type": rel_type,
        "weight": 1.0,
    }


def _make_storage(
    nodes_by_cluster: list[dict],
    edges_map: dict[str, list[dict]] | None = None,
    nodes_by_ids: list[dict] | None = None,
) -> MagicMock:
    """Return a mock storage that returns canned data."""
    storage = MagicMock()
    storage.get_nodes_by_cluster.return_value = nodes_by_cluster
    if edges_map is not None:
        def _get_edges_from(node_id, rel_types=None):
            return edges_map.get(node_id, [])
        storage.get_edges_from.side_effect = _get_edges_from
    else:
        storage.get_edges_from.return_value = []
    if nodes_by_ids is not None:
        storage.get_nodes_by_ids.return_value = nodes_by_ids
    else:
        storage.get_nodes_by_ids.return_value = []
    return storage


# ── Scenario 1: 3 nodes, 2 edges ──────────────────────────────────────────────


class TestThreeNodesWithEdges:
    """3 nodes + 2 intra-cluster edges → diagram contains all 3 names and 2 arrows."""

    def setup_method(self):
        nodes = [
            _node("n1", "WorkerPool"),
            _node("n2", "Worker"),
            _node("n3", "JobQueue"),
        ]
        edges_map = {
            "n1": [
                _edge("n1", "n2", "calls"),
                _edge("n1", "n3", "calls"),
            ],
            "n2": [],
            "n3": [],
        }
        storage = _make_storage(nodes, edges_map)
        self.dg = DiagramGenerator(storage)
        self.diagram = self.dg.generate_cluster_diagram(cluster_id=7, page_spec=None)

    def test_diagram_is_non_empty(self):
        assert self.diagram != ""

    def test_diagram_starts_with_mermaid_fence(self):
        assert self.diagram.startswith("```mermaid")

    def test_diagram_contains_graph_directive(self):
        assert "graph LR" in self.diagram or "graph TD" in self.diagram

    def test_diagram_contains_all_three_node_names(self):
        assert "WorkerPool" in self.diagram
        assert "Worker" in self.diagram
        assert "JobQueue" in self.diagram

    def test_diagram_contains_two_arrows(self):
        # Mermaid arrows are '-->'
        assert self.diagram.count("-->") == 2

    def test_diagram_ends_with_closing_fence(self):
        assert self.diagram.strip().endswith("```")


# ── Scenario 2: zero nodes ─────────────────────────────────────────────────────


class TestZeroNodes:
    """0 nodes → empty string."""

    def test_empty_storage_returns_empty_string(self):
        storage = _make_storage([])
        dg = DiagramGenerator(storage)
        result = dg.generate_cluster_diagram(cluster_id=3, page_spec=None)
        assert result == ""

    def test_none_storage_returns_empty_string(self):
        dg = DiagramGenerator(None)
        result = dg.generate_cluster_diagram(cluster_id=3, page_spec=None)
        assert result == ""


# ── Scenario 3: 20+ nodes — only architectural ones, max 15 ──────────────────


class TestManyNodes:
    """With 20+ candidate nodes, filter to architectural only, cap at 15."""

    def _make_nodes(self, count: int, is_arch_up_to: int) -> list[dict]:
        """Create `count` nodes where first `is_arch_up_to` are architectural."""
        return [
            _node(
                f"n{i}",
                f"Symbol{i}",
                is_architectural=1 if i < is_arch_up_to else 0,
            )
            for i in range(count)
        ]

    def test_non_architectural_nodes_excluded_when_over_limit(self):
        """With 20 nodes (5 arch, 15 non-arch), only arch nodes appear."""
        nodes = self._make_nodes(20, is_arch_up_to=5)
        storage = _make_storage(nodes)
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=1, page_spec=None)

        # Only arch nodes (Symbol0..Symbol4) should appear
        for i in range(5):
            assert f"Symbol{i}" in diagram
        # Non-arch nodes should NOT appear
        for i in range(5, 20):
            assert f"Symbol{i}" not in diagram

    def test_max_15_nodes_even_if_all_architectural(self):
        """With 25 architectural nodes, at most 15 appear in the diagram."""
        nodes = self._make_nodes(25, is_arch_up_to=25)
        storage = _make_storage(nodes)
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=1, page_spec=None)

        # Count occurrences of "Symbol" — each node name appears once
        # (either in a node definition or an edge)
        present = sum(1 for i in range(25) if f"Symbol{i}" in diagram)
        assert present <= 15

    def test_with_exactly_15_nodes_all_included(self):
        """Exactly 15 architectural nodes → all included."""
        nodes = self._make_nodes(15, is_arch_up_to=15)
        storage = _make_storage(nodes)
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=1, page_spec=None)

        for i in range(15):
            assert f"Symbol{i}" in diagram


# ── Scenario 4: isolated nodes (no edges) ────────────────────────────────────


class TestIsolatedNodes:
    """Nodes with no edges to other cluster nodes still appear as isolated nodes."""

    def test_isolated_nodes_appear_in_diagram(self):
        nodes = [
            _node("n1", "LoneClass"),
            _node("n2", "AnotherClass"),
        ]
        storage = _make_storage(nodes, edges_map={"n1": [], "n2": []})
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=2, page_spec=None)

        assert "LoneClass" in diagram
        assert "AnotherClass" in diagram

    def test_isolated_nodes_diagram_has_no_arrows(self):
        nodes = [_node("n1", "LoneClass")]
        storage = _make_storage(nodes, edges_map={"n1": []})
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=2, page_spec=None)
        assert "-->" not in diagram


# ── Scenario 5: cycles handled ────────────────────────────────────────────────


class TestCycles:
    """Self-loops and mutual calls are emitted as edges (Mermaid handles them)."""

    def test_self_loop_edge_emitted(self):
        nodes = [_node("n1", "RecursiveProcessor")]
        edges_map = {"n1": [_edge("n1", "n1", "calls")]}
        storage = _make_storage(nodes, edges_map)
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=5, page_spec=None)
        # Self-loop: RecursiveProcessor --> RecursiveProcessor
        assert "-->" in diagram

    def test_mutual_call_both_edges_present(self):
        nodes = [_node("n1", "Alpha"), _node("n2", "Beta")]
        edges_map = {
            "n1": [_edge("n1", "n2", "calls")],
            "n2": [_edge("n2", "n1", "calls")],
        }
        storage = _make_storage(nodes, edges_map)
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=6, page_spec=None)
        assert diagram.count("-->") == 2


# ── Scenario 6: cross-cluster edges excluded ──────────────────────────────────


class TestCrossClusterEdges:
    """Edges to nodes outside the cluster are excluded from the diagram."""

    def test_outgoing_edge_to_external_node_not_included(self):
        nodes = [_node("n1", "InCluster")]
        # Edge to "external" which is NOT in nodes_by_cluster
        edges_map = {
            "n1": [_edge("n1", "n_external", "calls")],
        }
        storage = _make_storage(nodes, edges_map)
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=8, page_spec=None)
        # n_external should not appear
        assert "n_external" not in diagram
        # Arrow count should be 0 (no intra-cluster edges)
        assert "-->" not in diagram


# ── Scenario 7: special chars in names ───────────────────────────────────────


class TestSpecialChars:
    """Names with spaces or special chars are escaped in node IDs."""

    def test_name_with_spaces_appears_in_diagram(self):
        nodes = [_node("n1", "My Worker Class")]
        storage = _make_storage(nodes, edges_map={"n1": []})
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=9, page_spec=None)
        # The display name should appear somewhere
        assert "My Worker Class" in diagram

    def test_node_id_has_no_spaces(self):
        """Node IDs in Mermaid must not contain raw spaces."""
        nodes = [_node("n1", "My Worker Class")]
        storage = _make_storage(nodes, edges_map={"n1": []})
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=9, page_spec=None)
        lines = [ln for ln in diagram.splitlines() if "My Worker Class" in ln]
        # Check that no line has unquoted spaces in node identifier
        for ln in lines:
            stripped = ln.strip()
            # A valid Mermaid node id with spaces is quoted: A["My Worker Class"]
            # If the name appears only inside quotes it's fine
            assert '["My Worker Class"]' in stripped or "My_Worker_Class" in stripped
