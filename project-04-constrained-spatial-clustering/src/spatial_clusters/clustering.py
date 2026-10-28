"""Capacity-constrained k-means and network cost metrics.

Capacitated k-means here is Lloyd's algorithm with the assignment step replaced
by an exact min-cost assignment. Each centre is repeated ``capacity`` times as a
slot, and the points are matched to slots with the Hungarian algorithm. That
guarantees no cluster exceeds its capacity at every iteration.

Network cost is approximated by the minimum spanning tree (MST) length of the
points in each cluster, which is the shortest total cable length needed to link
them. The diameter is the largest distance between two points in a cluster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class ClusterResult:
    """Labels, centres and iteration count for one clustering run."""

    method: str
    labels: np.ndarray
    centres: np.ndarray
    n_iter: int


def capacity_for(n_points: int, k: int, slack: float = 0.1) -> int:
    """Per-cluster capacity: an equal share of points, plus a fractional slack."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if n_points < k:
        raise ValueError("Need at least as many points as clusters.")
    if slack < 0:
        raise ValueError("slack must be non-negative.")
    return math.ceil(n_points / k * (1.0 + slack))


def assign_with_capacity(points: np.ndarray, centres: np.ndarray, capacity: int) -> np.ndarray:
    """Exact min-total-distance assignment of points to centres with a per-cluster cap."""
    n, k = len(points), len(centres)
    if capacity < 1:
        raise ValueError("capacity must be at least 1.")
    if n > k * capacity:
        raise ValueError(f"Infeasible: {n} points cannot fit in {k} clusters of capacity {capacity}.")
    cost = cdist(points, centres)
    slots = np.repeat(np.arange(k), capacity)  # slot -> cluster index
    rows, cols = linear_sum_assignment(cost[:, slots])
    labels = np.empty(n, dtype=int)
    labels[rows] = slots[cols]
    return labels


def capacitated_kmeans(
    points: np.ndarray,
    k: int,
    capacity: int | None = None,
    slack: float = 0.1,
    seed: int = 0,
    max_iter: int = 100,
) -> ClusterResult:
    """Lloyd iterations with a capacity-constrained assignment step.

    Initialised from ordinary k-means centres. Converges when the labels stop changing.
    Like k-means, this finds a local optimum, so the result depends on the start.
    """
    if capacity is None:
        capacity = capacity_for(len(points), k, slack)
    init = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(points).cluster_centers_
    centres = init.copy()
    labels = None
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        new_labels = assign_with_capacity(points, centres, capacity)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = points[labels == j]
            if len(members):
                centres[j] = members.mean(axis=0)  # empty clusters keep their old centre
    if np.bincount(labels, minlength=k).min() == 0:
        raise RuntimeError("A cluster ended up empty; try a different seed or larger slack.")
    return ClusterResult("capacitated_kmeans", labels, centres, n_iter)


def unconstrained_kmeans(points: np.ndarray, k: int, seed: int = 0) -> ClusterResult:
    """Ordinary k-means, for comparison."""
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(points)
    return ClusterResult("kmeans", model.labels_, model.cluster_centers_, int(model.n_iter_))


def mst_length_km(points: np.ndarray) -> float:
    """Length of the minimum spanning tree over the points, in kilometres."""
    if len(points) < 2:
        return 0.0
    dist = squareform(pdist(points))
    tree = minimum_spanning_tree(dist)
    return float(tree.sum()) / 1000.0


def diameter_km(points: np.ndarray) -> float:
    """Largest pairwise distance within the points, in kilometres."""
    if len(points) < 2:
        return 0.0
    return float(pdist(points).max()) / 1000.0


def compare_methods(points: np.ndarray, results: list[ClusterResult]) -> pd.DataFrame:
    """One row per method with balance, network cost and cohesion metrics."""
    rows = []
    for res in results:
        sizes = np.bincount(res.labels)
        mst = [mst_length_km(points[res.labels == j]) for j in range(len(sizes))]
        diam = [diameter_km(points[res.labels == j]) for j in range(len(sizes))]
        rows.append(
            {
                "method": res.method,
                "largest cluster": int(sizes.max()),
                "smallest cluster": int(sizes.min()),
                "total MST (km)": round(float(np.sum(mst)), 1),
                "largest diameter (km)": round(float(np.max(diam)), 1),
                "silhouette": round(float(silhouette_score(points, res.labels)), 3),
            }
        )
    return pd.DataFrame(rows).set_index("method")
