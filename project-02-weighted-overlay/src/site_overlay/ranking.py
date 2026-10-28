"""Top-site ranking and sensitivity of the ranking to the weights."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from site_overlay.grid import Grid
from site_overlay.scoring import weighted_overlay


def top_k_cells(score: np.ndarray, grid: Grid, k: int) -> pd.DataFrame:
    """Return the k highest-scoring valid cells, ranked from 1.

    NaN cells are excluded. If fewer than k valid cells exist, all are returned.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    flat = score.ravel()
    valid_idx = np.flatnonzero(np.isfinite(flat))
    order = valid_idx[np.argsort(-flat[valid_idx], kind="stable")][:k]
    rows, cols = np.unravel_index(order, grid.shape)
    xs, ys = grid.cell_centres()
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "row": rows,
            "col": cols,
            "x_m": xs[rows, cols],
            "y_m": ys[rows, cols],
            "score": flat[order],
        }
    )


def top_k_index_set(score: np.ndarray, k: int) -> set[int]:
    """Flat indices of the top-k valid cells, for set comparisons."""
    flat = score.ravel()
    valid_idx = np.flatnonzero(np.isfinite(flat))
    order = valid_idx[np.argsort(-flat[valid_idx], kind="stable")][:k]
    return set(order.tolist())


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard similarity of two sets. Returns 1.0 for two empty sets."""
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def weight_sweep(
    layers: dict[str, np.ndarray],
    base_weights: dict[str, float],
    vary: str,
    values: Iterable[float],
    k: int,
) -> pd.DataFrame:
    """Re-score with one weight varied and the rest rescaled proportionally.

    Compares each scenario with the base case on two measures:
    Spearman rank correlation over all valid cells, and Jaccard overlap of the top-k sets.
    """
    if vary not in base_weights:
        raise ValueError(f"'{vary}' is not a weight name.")
    others = [name for name in base_weights if name != vary]
    other_total = sum(base_weights[name] for name in others)
    if other_total <= 0:
        raise ValueError("Non-varied weights must have a positive total.")

    base_score = weighted_overlay(layers, base_weights)
    base_valid = np.isfinite(base_score.ravel())
    base_top = top_k_index_set(base_score, k)

    rows = []
    for w in values:
        if not 0.0 <= w <= 1.0:
            raise ValueError("Swept weight values must lie in [0, 1].")
        weights = {vary: float(w)}
        for name in others:
            weights[name] = (1.0 - w) * base_weights[name] / other_total
        scenario = weighted_overlay(layers, weights)
        valid = base_valid & np.isfinite(scenario.ravel())
        rho, _ = stats.spearmanr(base_score.ravel()[valid], scenario.ravel()[valid])
        rows.append(
            {
                vary: float(w),
                "spearman_vs_base": float(rho),
                "top_k_jaccard_vs_base": jaccard(base_top, top_k_index_set(scenario, k)),
            }
        )
    return pd.DataFrame(rows)
