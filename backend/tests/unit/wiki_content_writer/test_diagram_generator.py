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


def _class_node(
    node_id: str,
    symbol_name: str,
    symbol_type: str = "class",
    is_architectural: int = 1,
    source_text: str = "",
    parent_symbol: str | None = None,
    signature: str = "",
    parameters: str = "",
    return_type: str = "",
    macro_cluster: int = 1,
    micro_cluster: int = 0,
) -> dict[str, Any]:
    """Create a node dict matching the real storage API schema."""
    return {
        "node_id": node_id,
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "is_architectural": is_architectural,
        "source_text": source_text,
        "parent_symbol": parent_symbol,
        "signature": signature,
        "parameters": parameters,
        "return_type": return_type,
        "macro_cluster": macro_cluster,
        "micro_cluster": micro_cluster,
        # Legacy aliases so _node_name() works for both formats.
        "name": symbol_name,
    }


def _storage_edge(target_id: str, rel_type: str = "inheritance") -> dict[str, Any]:
    """Edge dict as returned by get_edges_from."""
    return {"target_id": target_id, "rel_type": rel_type}


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
        assert "```mermaid" in self.diagram

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

    def test_all_code_nodes_included_no_arch_filter(self):
        """All non-doc nodes appear — no is_architectural filter or count cap."""
        nodes = self._make_nodes(20, is_arch_up_to=5)
        storage = _make_storage(nodes)
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=1, page_spec=None)

        # All 20 nodes should appear (arch filter removed)
        for i in range(20):
            assert f"Symbol{i}" in diagram

    def test_all_architectural_nodes_included_no_cap(self):
        """All architectural nodes appear — no hard cap is applied."""
        nodes = self._make_nodes(20, is_arch_up_to=20)
        storage = _make_storage(nodes)
        dg = DiagramGenerator(storage)
        diagram = dg.generate_cluster_diagram(cluster_id=1, page_spec=None)

        present = sum(1 for i in range(20) if f"Symbol{i}" in diagram)
        assert present == 20


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


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 1: _select_nodes doc-type filtering
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelectNodesDocTypeFiltering:
    """_select_nodes must exclude document-type nodes from architecture diagrams."""

    def _doc_class_node(self, node_id: str, name: str, symbol_type: str) -> dict:
        return {
            "node_id": node_id,
            "symbol_name": name,
            "name": name,
            "symbol_type": symbol_type,
            "is_architectural": 1,
        }

    def test_markdown_document_excluded(self):
        from app.core.wiki_content_writer.diagram_generator import _select_nodes
        nodes = [self._doc_class_node("d1", "README", "markdown_document")]
        result = _select_nodes(nodes)
        assert result == []

    def test_config_document_excluded(self):
        from app.core.wiki_content_writer.diagram_generator import _select_nodes
        nodes = [self._doc_class_node("d1", "pyproject", "config_document")]
        result = _select_nodes(nodes)
        assert result == []

    def test_toml_document_excluded(self):
        from app.core.wiki_content_writer.diagram_generator import _select_nodes
        nodes = [self._doc_class_node("d1", "Cargo", "toml_document")]
        result = _select_nodes(nodes)
        assert result == []

    def test_module_doc_excluded(self):
        from app.core.wiki_content_writer.diagram_generator import _select_nodes
        nodes = [self._doc_class_node("d1", "MyModule", "module_doc")]
        result = _select_nodes(nodes)
        assert result == []

    def test_code_nodes_kept(self):
        from app.core.wiki_content_writer.diagram_generator import _select_nodes
        nodes = [
            _class_node("c1", "MyClass", "class"),
            _class_node("c2", "MyFunc", "function"),
        ]
        result = _select_nodes(nodes)
        assert len(result) == 2

    def test_mixed_nodes_only_code_returned(self):
        from app.core.wiki_content_writer.diagram_generator import _select_nodes
        nodes = [
            _class_node("c1", "MyClass", "class"),
            self._doc_class_node("d1", "README", "markdown_document"),
            _class_node("c2", "MyFunc", "function"),
            self._doc_class_node("d2", "Config", "config_document"),
        ]
        result = _select_nodes(nodes)
        names = [n["symbol_name"] for n in result]
        assert "MyClass" in names
        assert "MyFunc" in names
        assert "README" not in names
        assert "Config" not in names

    def test_generate_cluster_context_diagram_empty_when_all_doc_nodes(self):
        """generate_cluster_context_diagram returns '' when cluster has only doc nodes."""
        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = [
            {
                "node_id": "d1", "symbol_name": "README", "name": "README",
                "symbol_type": "markdown_document", "is_architectural": 1,
            },
            {
                "node_id": "d2", "symbol_name": "CHANGELOG", "name": "CHANGELOG",
                "symbol_type": "markdown_document", "is_architectural": 0,
            },
        ]
        storage.get_edges_from.return_value = []
        dg = DiagramGenerator(storage)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1, cluster_title="Docs", all_cluster_ids=[], cluster_titles={}
        )
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 2: parent_symbol format normalization
# ═══════════════════════════════════════════════════════════════════════════════


class TestParentSymbolNormalization:
    """_build_class_diagram must match dotted parent_symbol to class short name."""

    def _make_dg(self, nodes, edges_map=None):
        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = nodes
        storage.get_edges_from.side_effect = (
            lambda nid, rel_types=None: (edges_map or {}).get(nid, [])
        )
        return DiagramGenerator(storage)

    def test_dotted_parent_symbol_method_appears_in_class(self):
        """Method with parent_symbol='module.ClassName' should appear under ClassName."""
        nodes = [
            _class_node("c1", "WorkerPool", "class"),
            _class_node("c2", "JobQueue", "class"),
            # Method with dotted parent_symbol as stored in DB
            _class_node(
                "m1", "submit", "method",
                parent_symbol="worker.WorkerPool",
                signature="submit(job: Job) -> None",
            ),
        ]
        dg = self._make_dg(nodes)
        result = dg._build_class_diagram(cluster_id=1)
        # The method 'submit' must appear in the diagram under WorkerPool
        assert "submit" in result

    def test_simple_parent_symbol_still_works(self):
        """Method with simple (non-dotted) parent_symbol still matches."""
        nodes = [
            _class_node("c1", "Foo", "class"),
            _class_node("c2", "Bar", "class"),
            _class_node("m1", "do_it", "method", parent_symbol="Foo"),
        ]
        dg = self._make_dg(nodes)
        result = dg._build_class_diagram(cluster_id=1)
        assert "do_it" in result

    def test_deeply_dotted_parent_symbol_normalized(self):
        """Method with 'a.b.c.ClassName' parent_symbol resolves to 'ClassName'."""
        nodes = [
            _class_node("c1", "MyClass", "class"),
            _class_node("c2", "Other", "class"),
            _class_node(
                "m1", "run", "method",
                parent_symbol="pkg.subpkg.module.MyClass",
            ),
        ]
        dg = self._make_dg(nodes)
        result = dg._build_class_diagram(cluster_id=1)
        assert "run" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 3: _is_data_model only detects real DB models
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsDataModelStrict:
    """_is_data_model must NOT fire on BaseModel-only or @dataclass classes."""

    def test_pydantic_basemodel_not_detected(self):
        """Pydantic request/response schemas must NOT trigger erDiagram."""
        from app.core.wiki_content_writer.diagram_generator import _is_data_model
        node = _class_node(
            "m1", "LaunchResponse", "class",
            source_text="class LaunchResponse(BaseModel):\n    id: str\n    status: str",
        )
        assert _is_data_model(node) is False

    def test_dataclass_not_detected(self):
        """@dataclass DTOs must NOT trigger erDiagram."""
        from app.core.wiki_content_writer.diagram_generator import _is_data_model
        node = _class_node(
            "m1", "JobStatus", "class",
            source_text="@dataclass\nclass JobStatus:\n    id: str\n    state: str",
        )
        assert _is_data_model(node) is False

    def test_sqlalchemy_column_detected(self):
        from app.core.wiki_content_writer.diagram_generator import _is_data_model
        node = _class_node(
            "m1", "User", "class",
            source_text="class User(Base):\n    id = Column(Integer, primary_key=True)",
        )
        assert _is_data_model(node) is True

    def test_sqlalchemy_relationship_detected(self):
        from app.core.wiki_content_writer.diagram_generator import _is_data_model
        node = _class_node(
            "m1", "Post", "class",
            source_text="class Post(Base):\n    user_id = Column(Integer)\n    user = relationship('User')",
        )
        assert _is_data_model(node) is True

    def test_mapped_column_detected(self):
        from app.core.wiki_content_writer.diagram_generator import _is_data_model
        node = _class_node(
            "m1", "Article", "class",
            source_text="class Article(Base):\n    id: Mapped[int] = mapped_column(primary_key=True)",
        )
        assert _is_data_model(node) is True

    def test_declarative_base_detected(self):
        from app.core.wiki_content_writer.diagram_generator import _is_data_model
        node = _class_node(
            "m1", "Base", "class",
            source_text="class Base(DeclarativeBase):\n    pass",
        )
        assert _is_data_model(node) is True

    def test_db_model_detected(self):
        from app.core.wiki_content_writer.diagram_generator import _is_data_model
        node = _class_node(
            "m1", "User", "class",
            source_text="class User(db.Model):\n    id = db.Column(db.Integer)",
        )
        assert _is_data_model(node) is True

    def test_plain_class_not_detected(self):
        from app.core.wiki_content_writer.diagram_generator import _is_data_model
        node = _class_node(
            "m1", "Util", "class",
            source_text="class Util:\n    def run(self): pass",
        )
        assert _is_data_model(node) is False

    def test_generate_data_model_diagram_skips_basemodel_only(self):
        """generate_cluster_context_diagram skips Pydantic-only clusters."""
        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = [
            _class_node(
                "m1", "LaunchResponse", "class",
                source_text="class LaunchResponse(BaseModel):\n    id: str",
            ),
            _class_node(
                "m2", "JobStatus", "class",
                source_text="class JobStatus(BaseModel):\n    state: str",
            ),
        ]
        storage.get_edges_from.return_value = []
        dg = DiagramGenerator(storage)
        result = dg._build_data_model_diagram(cluster_id=1)
        assert result == ""

    def test_sqlalchemy_nodes_produce_er_diagram(self):
        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = [
            _class_node(
                "m1", "User", "class",
                source_text="class User(Base):\n    id = Column(Integer)",
            ),
            _class_node(
                "m2", "Post", "class",
                source_text="class Post(Base):\n    id = Column(Integer)\n    user = relationship('User')",
            ),
        ]
        storage.get_edges_from.return_value = []
        dg = DiagramGenerator(storage)
        result = dg._build_data_model_diagram(cluster_id=1)
        assert "erDiagram" in result


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: generate_cluster_context_diagram
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateClusterContextDiagram:
    """Unit tests for DiagramGenerator.generate_cluster_context_diagram."""

    def _make_dg(self, nodes, edges_map=None):
        def _side_effect(node_id, rel_types=None):
            if edges_map:
                return edges_map.get(node_id, [])
            return []

        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = nodes
        storage.get_edges_from.side_effect = _side_effect
        return DiagramGenerator(storage)

    # ── Intra-cluster nodes rendered in subgraph ───────────────────────────

    def test_connected_nodes_appear_in_subgraph(self):
        """Only nodes participating in edges appear — isolated nodes are omitted."""
        nodes = [
            _class_node("n1", "AuthService", "class"),
            _class_node("n2", "TokenParser", "class"),
        ]
        edges_map = {"n1": [_storage_edge("n2", "calls")], "n2": []}
        dg = self._make_dg(nodes, edges_map)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1, cluster_title="Auth", all_cluster_ids=[], cluster_titles={}
        )
        assert "subgraph" in result
        assert "AuthService" in result
        assert "TokenParser" in result

    def test_isolated_nodes_only_returns_empty(self):
        """Cluster with no edges produces an empty string — nothing to show."""
        nodes = [
            _class_node("n1", "AuthService", "class"),
            _class_node("n2", "TokenParser", "class"),
        ]
        dg = self._make_dg(nodes)  # no edges_map → no edges
        result = dg.generate_cluster_context_diagram(
            cluster_id=1, cluster_title="Auth", all_cluster_ids=[], cluster_titles={}
        )
        assert result == ""

    def test_diagram_has_mermaid_fence(self):
        nodes = [
            _class_node("n1", "A", "class"),
            _class_node("n2", "B", "class"),
        ]
        edges_map = {"n1": [_storage_edge("n2", "calls")], "n2": []}
        dg = self._make_dg(nodes, edges_map)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1, cluster_title="Test", all_cluster_ids=[], cluster_titles={}
        )
        assert "```mermaid" in result
        assert result.strip().endswith("```")

    def test_graph_td_header(self):
        """Layout switched to TD for readability."""
        nodes = [
            _class_node("n1", "X", "class"),
            _class_node("n2", "Y", "class"),
        ]
        edges_map = {"n1": [_storage_edge("n2", "calls")], "n2": []}
        dg = self._make_dg(nodes, edges_map)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1, cluster_title="X", all_cluster_ids=[], cluster_titles={}
        )
        assert "graph TD" in result

    # ── Empty cluster → empty string ──────────────────────────────────────

    def test_empty_cluster_returns_empty(self):
        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = []
        storage.get_edges_from.return_value = []
        dg = DiagramGenerator(storage)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1, cluster_title="Empty", all_cluster_ids=[], cluster_titles={}
        )
        assert result == ""

    def test_none_storage_returns_empty(self):
        dg = DiagramGenerator(None)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1, cluster_title="X", all_cluster_ids=[], cluster_titles={}
        )
        assert result == ""

    # ── Intra-cluster edges ────────────────────────────────────────────────

    def test_intra_cluster_edges_rendered(self):
        nodes = [
            _class_node("n1", "A", "class"),
            _class_node("n2", "B", "class"),
        ]
        edges_map = {
            "n1": [_storage_edge("n2", "calls")],
            "n2": [],
        }
        dg = self._make_dg(nodes, edges_map)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1, cluster_title="MyCluster", all_cluster_ids=[], cluster_titles={}
        )
        assert "-->" in result

    # ── Cross-cluster external nodes ───────────────────────────────────────

    def test_cross_cluster_edge_creates_external_node(self):
        # n1 is in cluster 1; n_ext is in cluster 2
        nodes_c1 = [_class_node("n1", "AuthService", "class")]
        nodes_c2 = [_class_node("n_ext", "WikiService", "class")]

        # Storage returns different nodes for different cluster IDs
        def _get_nodes_by_cluster(cluster_id):
            if cluster_id == 1:
                return nodes_c1
            elif cluster_id == 2:
                return nodes_c2
            return []

        edges_map = {
            "n1": [_storage_edge("n_ext", "calls")],
        }

        storage = MagicMock()
        storage.get_nodes_by_cluster.side_effect = _get_nodes_by_cluster
        storage.get_edges_from.side_effect = lambda nid, rel_types=None: edges_map.get(nid, [])

        dg = DiagramGenerator(storage)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1,
            cluster_title="Auth Service",
            all_cluster_ids=[1, 2],
            cluster_titles={1: "Auth Service", 2: "Wiki Service"},
        )
        # The external cluster title should appear as a node
        assert "Wiki Service" in result or "Wiki_Service" in result

    def test_cross_cluster_shown_as_single_node_not_individual_members(self):
        """External cluster appears as a single labeled node, not individual nodes."""
        nodes_c1 = [_class_node("n1", "AuthService", "class")]
        nodes_c2 = [
            _class_node("n_ext1", "WikiService", "class"),
            _class_node("n_ext2", "PageRenderer", "class"),
        ]

        def _get_nodes_by_cluster(cluster_id):
            if cluster_id == 1:
                return nodes_c1
            return nodes_c2

        edges_map = {
            "n1": [_storage_edge("n_ext1", "calls"), _storage_edge("n_ext2", "calls")],
        }

        storage = MagicMock()
        storage.get_nodes_by_cluster.side_effect = _get_nodes_by_cluster
        storage.get_edges_from.side_effect = lambda nid, rel_types=None: edges_map.get(nid, [])

        dg = DiagramGenerator(storage)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1,
            cluster_title="Auth",
            all_cluster_ids=[1, 2],
            cluster_titles={1: "Auth", 2: "Wiki"},
        )
        # WikiService individual nodes should NOT appear — only the cluster label "Wiki"
        assert "WikiService" not in result
        assert "PageRenderer" not in result

    # ── Storage error → graceful empty ────────────────────────────────────

    def test_storage_error_returns_empty(self):
        storage = MagicMock()
        storage.get_nodes_by_cluster.side_effect = RuntimeError("db down")
        dg = DiagramGenerator(storage)
        result = dg.generate_cluster_context_diagram(
            cluster_id=1, cluster_title="X", all_cluster_ids=[], cluster_titles={}
        )
        assert result == ""
