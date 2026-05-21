"""Unit tests for recluster_with_semantic_embeddings (#feat — semantic re-clustering).

Covers:
- Two small semantically-similar clusters are merged when both < 8 symbols.
- Two clusters that are both large (≥ 8 symbols) are NOT merged even if similar.
- Clusters with no embeddings are left unchanged.
- storage=None returns input list unchanged.
- High similarity but one large cluster → NOT merged.
- Return type is list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Minimal cluster stub ──────────────────────────────────────────────────────


@dataclass
class _FakeCluster:
    cluster_id: int
    kind: str = "code"
    dirs: list[str] = field(default_factory=list)
    artifacts: list[Any] = field(default_factory=list)
    total_artifacts: int = 0
    primary_languages: list[str] = field(default_factory=list)
    depth_range: tuple = (0, 0)


def _vec(values: list[float]) -> list[float]:
    return values


def _make_storage(
    cluster_node_map: dict[int, list[str]],
    embeddings: dict[str, list[float]],
) -> MagicMock:
    """Build a mock storage with cluster→node and node→embedding mappings."""
    storage = MagicMock()

    def _get_nodes_by_cluster(cid: int, limit: int = 1000):
        return [{"node_id": nid} for nid in cluster_node_map.get(cid, [])]

    def _get_embeddings_batch(node_ids: list[str]):
        return {nid: embeddings[nid] for nid in node_ids if nid in embeddings}

    storage.get_nodes_by_cluster.side_effect = _get_nodes_by_cluster
    storage.get_embeddings_batch.side_effect = _get_embeddings_batch
    return storage


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestReclusterImport:
    def test_module_importable(self):
        from app.core.semantic_clustering import recluster_with_semantic_embeddings  # noqa: F401


class TestReclusterNoop:
    def test_storage_none_returns_input(self):
        from app.core.semantic_clustering import recluster_with_semantic_embeddings

        clusters = [_FakeCluster(cluster_id=0), _FakeCluster(cluster_id=1)]
        result = recluster_with_semantic_embeddings(clusters, storage=None, embedding_fn=None)
        assert result is clusters

    def test_empty_clusters_returns_empty(self):
        from app.core.semantic_clustering import recluster_with_semantic_embeddings

        result = recluster_with_semantic_embeddings([], storage=MagicMock(), embedding_fn=None)
        assert result == []

    def test_no_embeddings_returns_input_unchanged(self):
        from app.core.semantic_clustering import recluster_with_semantic_embeddings

        storage = _make_storage({0: ["n0"], 1: ["n1"]}, embeddings={})  # no embeddings
        clusters = [_FakeCluster(cluster_id=0), _FakeCluster(cluster_id=1)]
        result = recluster_with_semantic_embeddings(clusters, storage=storage, embedding_fn=None)
        assert len(result) == 2


class TestReclusterSmallSimilarMerge:
    def test_two_small_similar_clusters_merged(self):
        """Two clusters with < 8 nodes each and high centroid similarity → merged."""
        from app.core.semantic_clustering import recluster_with_semantic_embeddings

        # Nearly identical embedding vectors
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.99, 0.14, 0.0]  # cosine ~0.98

        storage = _make_storage(
            cluster_node_map={0: ["n0"], 1: ["n1"]},
            embeddings={"n0": emb_a, "n1": emb_b},
        )

        c0 = _FakeCluster(cluster_id=0)
        c1 = _FakeCluster(cluster_id=1)
        result = recluster_with_semantic_embeddings(
            [c0, c1], storage=storage, embedding_fn=None, similarity_threshold=0.85
        )
        # One of the clusters should be removed (merged into the other)
        assert len(result) == 1

    def test_merged_cluster_has_lower_id(self):
        """When merging, the surviving cluster keeps the lower cluster_id."""
        from app.core.semantic_clustering import recluster_with_semantic_embeddings

        storage = _make_storage(
            cluster_node_map={5: ["n5"], 10: ["n10"]},
            embeddings={"n5": [1.0, 0.0], "n10": [0.99, 0.14]},
        )
        c5 = _FakeCluster(cluster_id=5)
        c10 = _FakeCluster(cluster_id=10)
        result = recluster_with_semantic_embeddings(
            [c5, c10], storage=storage, embedding_fn=None, similarity_threshold=0.85
        )
        assert len(result) == 1
        assert result[0].cluster_id == 5


class TestReclusterLargeClusterNotMerged:
    def test_large_cluster_not_merged_even_if_similar(self):
        """A cluster with ≥ 8 nodes is NOT merged even if centroid similarity is high."""
        from app.core.semantic_clustering import recluster_with_semantic_embeddings

        # 8 nodes in cluster 0 (too large), 1 node in cluster 1
        large_nodes = [f"n_big_{i}" for i in range(8)]
        large_embs = {nid: [1.0, 0.0, 0.0] for nid in large_nodes}
        small_embs = {"n_small": [0.99, 0.14, 0.0]}

        storage = _make_storage(
            cluster_node_map={0: large_nodes, 1: ["n_small"]},
            embeddings={**large_embs, **small_embs},
        )

        c0 = _FakeCluster(cluster_id=0)
        c1 = _FakeCluster(cluster_id=1)
        result = recluster_with_semantic_embeddings(
            [c0, c1], storage=storage, embedding_fn=None, similarity_threshold=0.85
        )
        # Both clusters remain — no merge because cluster 0 is large
        assert len(result) == 2

    def test_both_large_not_merged(self):
        """Two large clusters are never merged."""
        from app.core.semantic_clustering import recluster_with_semantic_embeddings

        nodes_a = [f"na_{i}" for i in range(8)]
        nodes_b = [f"nb_{i}" for i in range(8)]
        embs = {nid: [1.0, 0.0] for nid in nodes_a + nodes_b}

        storage = _make_storage(
            cluster_node_map={0: nodes_a, 1: nodes_b},
            embeddings=embs,
        )

        result = recluster_with_semantic_embeddings(
            [_FakeCluster(cluster_id=0), _FakeCluster(cluster_id=1)],
            storage=storage,
            embedding_fn=None,
            similarity_threshold=0.85,
        )
        assert len(result) == 2


class TestReclusterDissimilarNotMerged:
    def test_dissimilar_small_clusters_not_merged(self):
        """Two small clusters with low cosine similarity are NOT merged."""
        from app.core.semantic_clustering import recluster_with_semantic_embeddings

        storage = _make_storage(
            cluster_node_map={0: ["n0"], 1: ["n1"]},
            embeddings={"n0": [1.0, 0.0, 0.0], "n1": [0.0, 1.0, 0.0]},  # orthogonal → sim=0
        )

        result = recluster_with_semantic_embeddings(
            [_FakeCluster(cluster_id=0), _FakeCluster(cluster_id=1)],
            storage=storage,
            embedding_fn=None,
            similarity_threshold=0.85,
        )
        assert len(result) == 2


class TestReclusterReturnType:
    def test_returns_list(self):
        from app.core.semantic_clustering import recluster_with_semantic_embeddings

        storage = _make_storage({0: ["n0"]}, {"n0": [1.0, 0.0]})
        result = recluster_with_semantic_embeddings(
            [_FakeCluster(cluster_id=0)], storage=storage, embedding_fn=None
        )
        assert isinstance(result, list)
