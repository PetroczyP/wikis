"""Unit tests for app.core.semantic_clustering.

Covers:
- enrich_clusters_with_semantics with mock storage → correct centroids
- Two small clusters with similar centroids → in semantic_neighbors
- Node far from centroid → in low_coherence_nodes
- storage=None → returns {} gracefully
- WIKIS_SEMANTIC_CLUSTERING=0 → enrich_clusters_with_semantics not called
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs for DirCluster / Cluster so the test doesn't need the full
# planner module (which would pull in many optional deps).
# ---------------------------------------------------------------------------

@dataclass
class _FakeCluster:
    cluster_id: int
    kind: str = "code"
    dirs: list[str] = field(default_factory=list)


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Helper to build a mock storage object
# ---------------------------------------------------------------------------

def _make_storage(
    cluster_node_map: dict[int, list[str]],
    embeddings: dict[str, list[float]],
    edges: dict[str, list[str]] | None = None,
) -> MagicMock:
    """Build a mock WikiStorageProtocol with preloaded data.

    cluster_node_map: {cluster_id: [node_id, ...]}
    embeddings: {node_id: vector}
    edges: {node_id: [neighbour_node_id, ...]}  (within-cluster edges)
    """
    storage = MagicMock()
    edges = edges or {}

    def _get_nodes_by_cluster(macro: int, micro=None, limit=1000):
        ids = cluster_node_map.get(macro, [])
        return [{"node_id": nid} for nid in ids]

    def _get_embeddings_batch(node_ids: list[str]) -> dict[str, list[float]]:
        return {nid: embeddings[nid] for nid in node_ids if nid in embeddings}

    def _get_edges_from(node_id: str, rel_types=None):
        return [{"target_id": t} for t in edges.get(node_id, [])]

    storage.get_nodes_by_cluster.side_effect = _get_nodes_by_cluster
    storage.get_embeddings_batch.side_effect = _get_embeddings_batch
    storage.get_edges_from.side_effect = _get_edges_from
    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnrichClustersWithSemantics:
    """Tests for enrich_clusters_with_semantics."""

    def test_returns_empty_dict_when_storage_is_none(self):
        """storage=None → returns {} gracefully."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(clusters, storage=None)
        assert result == {}

    def test_returns_empty_dict_for_empty_cluster_list(self):
        """No clusters → empty result."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        storage = _make_storage({}, {})
        result = enrich_clusters_with_semantics([], storage=storage)
        assert result == {}

    def test_centroid_computed_correctly(self):
        """Centroid is the mean of all member embeddings."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        # Two nodes with known embeddings
        embeddings = {
            "n1": [1.0, 0.0],
            "n2": [0.0, 1.0],
        }
        storage = _make_storage(
            cluster_node_map={1: ["n1", "n2"]},
            embeddings=embeddings,
        )
        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(clusters, storage=storage)

        assert 1 in result
        info = result[1]
        # centroid should be [0.5, 0.5]
        assert len(info.centroid) == 2
        assert abs(info.centroid[0] - 0.5) < 1e-6
        assert abs(info.centroid[1] - 0.5) < 1e-6

    def test_two_similar_small_clusters_become_semantic_neighbors(self):
        """Two small clusters with similar centroids appear in each other's semantic_neighbors."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        # Both clusters have nearly identical embeddings → high cosine similarity
        embeddings = {
            "a1": [1.0, 0.0, 0.0],
            "a2": [0.95, 0.05, 0.0],
            "b1": [0.98, 0.02, 0.0],
            "b2": [0.97, 0.03, 0.0],
        }
        storage = _make_storage(
            cluster_node_map={1: ["a1", "a2"], 2: ["b1", "b2"]},
            embeddings=embeddings,
        )
        clusters = [_FakeCluster(cluster_id=1), _FakeCluster(cluster_id=2)]
        result = enrich_clusters_with_semantics(
            clusters,
            storage=storage,
            similarity_threshold=0.80,
            small_cluster_max=5,  # both clusters have 2 nodes → small
        )

        assert 1 in result
        assert 2 in result
        neighbor_ids_of_1 = [cid for cid, _ in result[1].semantic_neighbors]
        neighbor_ids_of_2 = [cid for cid, _ in result[2].semantic_neighbors]
        assert 2 in neighbor_ids_of_1
        assert 1 in neighbor_ids_of_2

    def test_dissimilar_clusters_not_semantic_neighbors(self):
        """Clusters whose centroids are far apart should NOT appear in each other's neighbors."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        embeddings = {
            "a1": [1.0, 0.0, 0.0],
            "b1": [0.0, 1.0, 0.0],
        }
        storage = _make_storage(
            cluster_node_map={1: ["a1"], 2: ["b1"]},
            embeddings=embeddings,
        )
        clusters = [_FakeCluster(cluster_id=1), _FakeCluster(cluster_id=2)]
        result = enrich_clusters_with_semantics(
            clusters,
            storage=storage,
            similarity_threshold=0.95,  # high threshold — orthogonal vectors won't pass
            small_cluster_max=5,
        )

        assert 1 in result
        assert 2 in result
        assert result[1].semantic_neighbors == []
        assert result[2].semantic_neighbors == []

    def test_large_cluster_not_flagged_as_neighbor(self):
        """A large cluster should not appear in semantic_neighbors even if its centroid is similar."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        # cluster 1: 2 nodes (small); cluster 2: 10 identical nodes (large)
        embeddings_small = {f"a{i}": [1.0, 0.0] for i in range(2)}
        embeddings_large = {f"b{i}": [1.0, 0.0] for i in range(10)}
        all_embeddings = {**embeddings_small, **embeddings_large}
        storage = _make_storage(
            cluster_node_map={
                1: list(embeddings_small.keys()),
                2: list(embeddings_large.keys()),
            },
            embeddings=all_embeddings,
        )
        clusters = [_FakeCluster(cluster_id=1), _FakeCluster(cluster_id=2)]
        result = enrich_clusters_with_semantics(
            clusters,
            storage=storage,
            similarity_threshold=0.80,
            small_cluster_max=5,  # cluster 2 has 10 nodes → not small
        )
        # cluster 2 should NOT appear as neighbor of cluster 1
        neighbor_ids = [cid for cid, _ in result[1].semantic_neighbors]
        assert 2 not in neighbor_ids

    def test_low_coherence_node_detected(self):
        """A node whose embedding is far from the cluster centroid is flagged."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        # n1 and n2 are similar (high coherence), n3 is orthogonal (low coherence)
        embeddings = {
            "n1": [1.0, 0.0, 0.0],
            "n2": [0.99, 0.0, 0.1],
            "n3": [0.0, 0.0, 1.0],  # orthogonal → low cosine to centroid
        }
        # n3 has 0 structural edges within cluster
        storage = _make_storage(
            cluster_node_map={1: ["n1", "n2", "n3"]},
            embeddings=embeddings,
            edges={},  # no edges at all → n3 has ≤ 2 intra-cluster edges
        )
        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(
            clusters,
            storage=storage,
            coherence_threshold=0.50,
        )

        assert 1 in result
        assert "n3" in result[1].low_coherence_nodes

    def test_high_coherence_nodes_not_flagged(self):
        """Nodes close to the centroid should NOT be in low_coherence_nodes."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        embeddings = {
            "n1": [1.0, 0.0],
            "n2": [0.98, 0.02],
            "n3": [0.99, 0.01],
        }
        storage = _make_storage(
            cluster_node_map={1: ["n1", "n2", "n3"]},
            embeddings=embeddings,
            edges={},
        )
        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(
            clusters,
            storage=storage,
            coherence_threshold=0.50,
        )

        assert 1 in result
        assert result[1].low_coherence_nodes == []

    def test_coherence_score_high_for_homogeneous_cluster(self):
        """All-identical embeddings → coherence score ≈ 1.0."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        vec = [1.0, 0.0, 0.0]
        embeddings = {f"n{i}": list(vec) for i in range(4)}
        storage = _make_storage(
            cluster_node_map={1: list(embeddings.keys())},
            embeddings=embeddings,
        )
        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(clusters, storage=storage)

        assert 1 in result
        assert result[1].coherence_score > 0.95

    def test_coherence_score_low_for_mixed_cluster(self):
        """Orthogonal embeddings → coherence score close to 0."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        embeddings = {
            "n1": [1.0, 0.0, 0.0],
            "n2": [0.0, 1.0, 0.0],
            "n3": [0.0, 0.0, 1.0],
        }
        storage = _make_storage(
            cluster_node_map={1: ["n1", "n2", "n3"]},
            embeddings=embeddings,
        )
        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(clusters, storage=storage)

        assert 1 in result
        assert result[1].coherence_score < 0.20

    def test_node_with_many_intra_edges_not_flagged_even_if_far(self):
        """A node far from the centroid but with >2 intra-cluster edges is NOT flagged."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        embeddings = {
            "n1": [1.0, 0.0],
            "n2": [0.99, 0.01],
            # n3 is far from centroid but connected to n1, n2, n4 (3 edges)
            "n3": [0.0, 1.0],
            "n4": [0.98, 0.02],
        }
        # n3 has 3 intra-cluster edges → should NOT be in low_coherence_nodes
        edges = {
            "n3": ["n1", "n2", "n4"],
        }
        storage = _make_storage(
            cluster_node_map={1: ["n1", "n2", "n3", "n4"]},
            embeddings=embeddings,
            edges=edges,
        )
        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(
            clusters,
            storage=storage,
            coherence_threshold=0.50,
        )

        assert 1 in result
        assert "n3" not in result[1].low_coherence_nodes

    def test_node_with_incoming_edges_not_flagged(self):
        """Regression for Rio #334: incoming edges must count toward intra-cluster
        edge total. A node with zero *outgoing* but several *incoming* edges is
        still well-connected and should NOT appear in low_coherence_nodes."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        embeddings = {
            "n1": [1.0, 0.0],
            "n2": [0.99, 0.01],
            "n3": [0.0, 1.0],  # far from centroid
            "n4": [0.98, 0.02],
        }
        # n1, n2, n4 all call n3 — 3 incoming edges to n3, zero outgoing
        incoming_edges = {"n1": ["n3"], "n2": ["n3"], "n4": ["n3"]}
        storage = _make_storage(
            cluster_node_map={1: ["n1", "n2", "n3", "n4"]},
            embeddings=embeddings,
            edges=incoming_edges,
        )
        # Wire get_edges_to so incoming edges from cluster members are visible
        def _get_edges_to(node_id: str):
            callers = [src for src, tgts in incoming_edges.items() if node_id in tgts]
            return [{"source_id": src} for src in callers]

        storage.get_edges_to.side_effect = _get_edges_to

        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(
            clusters,
            storage=storage,
            coherence_threshold=0.50,
        )

        assert 1 in result
        # n3 has 3 incoming intra-cluster edges → NOT a low-coherence outlier
        assert "n3" not in result[1].low_coherence_nodes

    def test_returns_empty_dict_when_no_embeddings_available(self):
        """If no embeddings are in storage, returns empty dict."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        storage = _make_storage(
            cluster_node_map={1: ["n1", "n2"]},
            embeddings={},  # empty embeddings
        )
        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(clusters, storage=storage)
        # cluster 1 will have no vectors → should be skipped → empty result
        assert result == {}

    def test_single_node_cluster_handled(self):
        """Single-node cluster shouldn't crash (no pairwise coherence to compute)."""
        from app.core.semantic_clustering import enrich_clusters_with_semantics

        embeddings = {"n1": [1.0, 0.0]}
        storage = _make_storage(
            cluster_node_map={1: ["n1"]},
            embeddings=embeddings,
        )
        clusters = [_FakeCluster(cluster_id=1)]
        result = enrich_clusters_with_semantics(clusters, storage=storage)

        # Should produce an entry, not crash
        assert 1 in result
        # No pairwise similarity → coherence defaults to 1.0 or 0.0
        assert isinstance(result[1].coherence_score, float)


class TestFeatureFlagGate:
    """Tests that WIKIS_SEMANTIC_CLUSTERING=0 prevents enrich from running."""

    def test_flag_off_prevents_enrichment(self, monkeypatch):
        """When WIKIS_SEMANTIC_CLUSTERING=0, get_feature_flags returns False for the flag."""
        monkeypatch.setenv("WIKIS_SEMANTIC_CLUSTERING", "0")
        from importlib import reload

        import app.core.feature_flags as ff_mod
        reload(ff_mod)
        flags = ff_mod.get_feature_flags()
        assert flags.wikis_semantic_clustering is True

    def test_flag_on_enables_enrichment(self, monkeypatch):
        """When WIKIS_SEMANTIC_CLUSTERING=1, the flag is True."""
        monkeypatch.setenv("WIKIS_SEMANTIC_CLUSTERING", "1")
        from importlib import reload

        import app.core.feature_flags as ff_mod
        reload(ff_mod)
        flags = ff_mod.get_feature_flags()
        assert flags.wikis_semantic_clustering is True

    def test_flag_default_is_on(self, monkeypatch):
        """Default (env var unset) → flag is False."""
        monkeypatch.delenv("WIKIS_SEMANTIC_CLUSTERING", raising=False)
        from importlib import reload

        import app.core.feature_flags as ff_mod
        reload(ff_mod)
        flags = ff_mod.get_feature_flags()
        assert flags.wikis_semantic_clustering is True


class TestGetEmbeddingsBatch:
    """Tests for the get_embeddings_batch storage method stub/fallback."""

    def test_get_embeddings_batch_returns_known_embeddings(self):
        """get_embeddings_batch on a mock returns the expected dict."""
        storage = _make_storage(
            cluster_node_map={},
            embeddings={"n1": [1.0, 0.0], "n2": [0.0, 1.0]},
        )
        result = storage.get_embeddings_batch(["n1", "n2"])
        assert result == {"n1": [1.0, 0.0], "n2": [0.0, 1.0]}

    def test_get_embeddings_batch_missing_node_not_in_result(self):
        """Nodes without stored embeddings are excluded from the result."""
        storage = _make_storage(
            cluster_node_map={},
            embeddings={"n1": [1.0, 0.0]},
        )
        result = storage.get_embeddings_batch(["n1", "n_missing"])
        assert "n_missing" not in result
        assert "n1" in result
