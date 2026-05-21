"""Semantic enrichment passes on top of structural Leiden clustering.

This module implements:

  * **Pass 2 — Semantic enrichment**: compute per-cluster centroid vectors
    from stored embeddings and flag semantically related clusters as merge
    candidates (``semantic_neighbors``).

  * **Pass 3 — Embedding validation (coherence check)**: compute per-cluster
    internal coherence score and flag nodes that are likely misfits
    (``low_coherence_nodes``).

Both passes are purely **observational** — they annotate ``DirCluster``
objects with extra metadata but do **not** reassign symbols between clusters.
Gated by ``WIKIS_SEMANTIC_CLUSTERING`` feature flag (default off).

Usage::

    from app.core.semantic_clustering import enrich_clusters_with_semantics

    semantic_info = enrich_clusters_with_semantics(
        skeleton.clusters,
        storage=storage,
    )
    # semantic_info: {cluster_id: SemanticClusterInfo}
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class SemanticClusterInfo:
    """Semantic annotation for a single structural cluster.

    Attributes:
        cluster_id: The macro cluster ID this info relates to.
        centroid: Mean of all member node embedding vectors.
        coherence_score: Mean pairwise cosine similarity within the cluster.
            Range [0, 1]; higher is more coherent.  Returns 1.0 for
            single-node clusters (trivially coherent).
        semantic_neighbors: List of ``(other_cluster_id, similarity)`` pairs
            where the two clusters' centroids exceed ``similarity_threshold``
            and *both* are small (≤ ``small_cluster_max``).
        low_coherence_nodes: Node IDs whose individual cosine distance to the
            cluster centroid is below ``coherence_threshold`` AND that have
            ≤ 2 structural edges within the cluster.
    """

    cluster_id: int
    centroid: list[float]
    coherence_score: float
    semantic_neighbors: list[tuple[int, float]] = field(default_factory=list)
    low_coherence_nodes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal vector math
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 for zero-length vectors to avoid division by zero.
    """
    try:
        import numpy as np  # noqa: PLC0415
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    except ImportError:
        # Pure-Python fallback for environments without numpy.
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)


def _mean_vector(vecs: list[list[float]]) -> list[float]:
    """Compute the element-wise mean of a list of vectors.

    All vectors must have the same dimension.  Returns an empty list for
    an empty input.
    """
    if not vecs:
        return []
    try:
        import numpy as np  # noqa: PLC0415
        arr = np.array(vecs, dtype=float)
        return arr.mean(axis=0).tolist()
    except ImportError:
        dim = len(vecs[0])
        totals = [0.0] * dim
        n = len(vecs)
        for v in vecs:
            for i, x in enumerate(v):
                totals[i] += x
        return [t / n for t in totals]


def _intra_cluster_edge_count(node_id: str, cluster_node_ids: set[str], storage: Any) -> int:
    """Count intra-cluster edges (both outgoing and incoming) for ``node_id``.

    A node with many incoming edges from cluster siblings but no outgoing edges
    is still well-connected; counting only one direction would incorrectly
    flag it as an outlier.  We union both sets before counting.
    """
    count = 0
    try:
        for e in storage.get_edges_from(node_id):
            if e.get("target_id") in cluster_node_ids:
                count += 1
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_edges_from(%r) failed: %s", node_id, exc)
    try:
        for e in storage.get_edges_to(node_id):
            if e.get("source_id") in cluster_node_ids:
                count += 1
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_edges_to(%r) failed: %s", node_id, exc)
    return count


# ---------------------------------------------------------------------------
# Batch embedding fetch with fallback
# ---------------------------------------------------------------------------

def _fetch_embeddings(node_ids: list[str], storage: Any) -> dict[str, list[float]]:
    """Fetch embeddings for ``node_ids`` from ``storage``.

    Prefers ``storage.get_embeddings_batch`` (a single bulk query) and falls
    back to iterating ``storage.get_embedding_by_id`` per node when the batch
    method is absent.  Missing embeddings are silently excluded.
    """
    if not node_ids:
        return {}

    # Try the batch method first.
    batch_fn = getattr(storage, "get_embeddings_batch", None)
    if callable(batch_fn):
        try:
            result = batch_fn(node_ids)
            if result is not None:
                return {k: v for k, v in result.items() if v}
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_embeddings_batch failed, falling back to per-node: %s", exc)

    # Fallback: per-node lookup.
    single_fn = getattr(storage, "get_embedding_by_id", None)
    if not callable(single_fn):
        return {}

    result = {}
    for nid in node_ids:
        try:
            vec = single_fn(nid)
            if vec:
                result[nid] = vec
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_embedding_by_id(%r) failed: %s", nid, exc)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_clusters_with_semantics(
    clusters: list,  # list[Cluster | DirCluster]
    storage: Any,    # WikiStorageProtocol | None
    similarity_threshold: float = 0.80,
    coherence_threshold: float = 0.50,
    small_cluster_max: int = 5,
) -> dict[int, SemanticClusterInfo]:
    """Run Pass 2 (semantic enrichment) + Pass 3 (coherence check).

    Both passes are **observational only** — they annotate clusters with
    semantic metadata but do not mutate cluster membership.

    Args:
        clusters: List of structural clusters produced by Louvain / Leiden.
            Each element must expose a ``.cluster_id`` attribute.
        storage: ``WikiStorageProtocol`` instance used to fetch node IDs and
            their stored embeddings.  Returns ``{}`` immediately when ``None``
            (storage not yet available or feature disabled upstream).
        similarity_threshold: Minimum centroid-to-centroid cosine similarity
            for two clusters to be flagged as merge candidates.  Both clusters
            must also qualify as "small" (≤ ``small_cluster_max`` symbols).
        coherence_threshold: A node whose cosine similarity to its cluster's
            centroid falls below this value AND whose intra-cluster edge count
            is ≤ 2 is added to ``low_coherence_nodes``.
        small_cluster_max: Maximum number of member nodes for a cluster to
            qualify as "small" when evaluating merge candidacy.

    Returns:
        ``{cluster_id: SemanticClusterInfo}`` for every cluster that had at
        least one stored embedding.  Clusters with no embeddings are omitted.
        Returns ``{}`` when ``storage`` is ``None`` or ``clusters`` is empty.
    """
    if storage is None or not clusters:
        return {}

    # ── Pass 2a: collect node IDs and embeddings per cluster ──────────────
    cluster_node_ids: dict[int, list[str]] = {}
    for cluster in clusters:
        cid = cluster.cluster_id
        try:
            rows = storage.get_nodes_by_cluster(cid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SEMANTIC] get_nodes_by_cluster(%d) failed: %s", cid, exc)
            rows = []
        cluster_node_ids[cid] = [r["node_id"] for r in rows if r.get("node_id")]

    # Bulk-fetch embeddings for all nodes across all clusters.
    all_node_ids: list[str] = []
    for ids in cluster_node_ids.values():
        all_node_ids.extend(ids)

    all_embeddings = _fetch_embeddings(all_node_ids, storage)

    if not all_embeddings:
        logger.debug("[SEMANTIC] no embeddings available — skipping semantic enrichment")
        return {}

    # ── Pass 2b: compute per-cluster centroids ────────────────────────────
    centroids: dict[int, list[float]] = {}
    cluster_vecs: dict[int, dict[str, list[float]]] = {}  # cid → {node_id: vec}
    for cid, node_ids in cluster_node_ids.items():
        vecs_for_cluster = {
            nid: all_embeddings[nid] for nid in node_ids if nid in all_embeddings
        }
        if not vecs_for_cluster:
            continue
        cluster_vecs[cid] = vecs_for_cluster
        centroids[cid] = _mean_vector(list(vecs_for_cluster.values()))

    if not centroids:
        return {}

    # ── Pass 2c: find semantic neighbors (both clusters must be small) ────
    cids = list(centroids.keys())
    neighbor_map: dict[int, list[tuple[int, float]]] = {cid: [] for cid in cids}

    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            cid_a = cids[i]
            cid_b = cids[j]
            size_a = len(cluster_vecs[cid_a])
            size_b = len(cluster_vecs[cid_b])
            # Both must be small.
            if size_a > small_cluster_max or size_b > small_cluster_max:
                continue
            sim = _cosine_similarity(centroids[cid_a], centroids[cid_b])
            if sim >= similarity_threshold:
                neighbor_map[cid_a].append((cid_b, sim))
                neighbor_map[cid_b].append((cid_a, sim))

    # ── Pass 3: coherence check ───────────────────────────────────────────
    result: dict[int, SemanticClusterInfo] = {}
    for cid in cids:
        vecs = cluster_vecs[cid]
        centroid = centroids[cid]
        node_ids_in_cluster = list(vecs.keys())
        cluster_id_set = set(cluster_node_ids[cid])

        # Pass 3a: pairwise cosine similarities for coherence score.
        if len(node_ids_in_cluster) == 1:
            coherence_score = 1.0
        else:
            pairwise: list[float] = []
            vec_list = list(vecs.values())
            for ii in range(len(vec_list)):
                for jj in range(ii + 1, len(vec_list)):
                    pairwise.append(_cosine_similarity(vec_list[ii], vec_list[jj]))
            coherence_score = sum(pairwise) / len(pairwise) if pairwise else 0.0

        # Pass 3b: flag low-coherence nodes.
        low_coherence: list[str] = []
        for nid, vec in vecs.items():
            sim_to_centroid = _cosine_similarity(vec, centroid)
            if sim_to_centroid < coherence_threshold:
                intra_edges = _intra_cluster_edge_count(nid, cluster_id_set, storage)
                if intra_edges <= 2:
                    low_coherence.append(nid)

        result[cid] = SemanticClusterInfo(
            cluster_id=cid,
            centroid=centroid,
            coherence_score=coherence_score,
            semantic_neighbors=neighbor_map.get(cid, []),
            low_coherence_nodes=low_coherence,
        )

    logger.info(
        "[SEMANTIC] enrichment complete: %d clusters annotated, "
        "%d total neighbor pairs, %d total low-coherence nodes",
        len(result),
        sum(len(info.semantic_neighbors) for info in result.values()),
        sum(len(info.low_coherence_nodes) for info in result.values()),
    )
    return result
