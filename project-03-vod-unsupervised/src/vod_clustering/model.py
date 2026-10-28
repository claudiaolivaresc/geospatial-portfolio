"""Unsupervised clustering pipeline, evaluation and baseline.

Discipline: the scaler and PCA are fitted on training pixels only. Test pixels
are assigned with ``predict`` and never influence the fitted transforms. Land
cover labels are used only for evaluation, never for fitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(n_clusters: int, n_components: int = 2, seed: int = 0) -> Pipeline:
    """Standardise, project with PCA, then cluster with k-means."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)),
        ]
    )


def choose_k(
    X_train: np.ndarray, ks: range | list[int], seed: int = 0, sample_size: int = 3000
) -> dict[int, float]:
    """Silhouette score on a training sample for each candidate k. Higher is better."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(sample_size, len(X_train)), replace=False)
    scores = {}
    for k in ks:
        pipe = build_pipeline(k, seed=seed).fit(X_train)
        Z = pipe[:-1].transform(X_train[idx])  # the PCA space the clusters live in
        labels = pipe.predict(X_train[idx])
        scores[int(k)] = float(silhouette_score(Z, labels))
    return scores


def evaluate_assignments(
    labels_pred: np.ndarray, labels_true: np.ndarray, Z_test: np.ndarray
) -> dict[str, float]:
    """Compare test assignments with land-cover labels (evaluation only).

    Args:
        labels_pred: cluster assignments for test pixels.
        labels_true: land-cover classes for the same pixels. Used only here.
        Z_test: test pixels in the PCA space the clusters were fitted in.

    Adjusted Rand index is 1 for perfect agreement and about 0 for random assignment.
    """
    ari = adjusted_rand_score(labels_true, labels_pred)
    n_clusters = len(np.unique(labels_pred))
    sil = (
        silhouette_score(Z_test, labels_pred, sample_size=3000, random_state=0)
        if n_clusters > 1
        else np.nan
    )
    return {"ARI_vs_land_cover": float(ari), "silhouette_test": float(sil), "n_clusters_used": n_clusters}


def baseline_quantile_labels(
    X_train: np.ndarray, X_test: np.ndarray, band: int, k: int
) -> np.ndarray:
    """Baseline: bin one band into k quantile classes, with edges fitted on training data."""
    edges = np.quantile(X_train[:, band], np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(X_test[:, band], edges)


def seed_stability(X_train: np.ndarray, X_test: np.ndarray, k: int, seeds: tuple[int, int]) -> float:
    """Adjusted Rand index between test assignments from two k-means seeds.

    High values mean the partition is reproducible, not an artefact of initialisation.
    """
    a = build_pipeline(k, seed=seeds[0]).fit(X_train).predict(X_test)
    b = build_pipeline(k, seed=seeds[1]).fit(X_train).predict(X_test)
    return float(adjusted_rand_score(a, b))
