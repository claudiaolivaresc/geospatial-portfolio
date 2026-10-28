"""Normalisation and weighted-overlay scoring.

Every layer is first scaled to [0, 1], with higher meaning more suitable,
then combined as a weighted sum. Cells masked as NaN in any input are NaN in
the output and are never ranked.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def min_max(values: np.ndarray) -> np.ndarray:
    """Scale finite values to [0, 1] using the observed range. NaN is preserved.

    A constant layer maps to zeros, because it does not discriminate between cells.
    The scale depends on the study-area extent, so scores are only comparable
    within the same extent.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Layer has no finite values to normalise.")
    lo = values[finite].min()
    hi = values[finite].max()
    out = np.full(values.shape, np.nan)
    if hi == lo:
        out[finite] = 0.0
    else:
        out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def proximity_score(distance_m: np.ndarray) -> np.ndarray:
    """Higher score for shorter distance. Distances must be in metres."""
    return 1.0 - min_max(distance_m)


def validate_weights(weights: Mapping[str, float], tol: float = 1e-9) -> None:
    """Raise ValueError unless weights are non-negative and sum to 1."""
    if not weights:
        raise ValueError("At least one weight is required.")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Weights must be non-negative.")
    total = sum(weights.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1, got {total:.6f}.")


def weighted_overlay(
    layers: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    """Combine normalised layers into one suitability score.

    Args:
        layers: name -> 2-D array with values in [0, 1] (NaN where excluded).
        weights: name -> weight. Keys must match ``layers`` and sum to 1.

    Returns:
        2-D score array. A cell is NaN if any input is NaN there.
    """
    validate_weights(weights)
    if set(layers) != set(weights):
        raise ValueError(
            f"Layer and weight names differ: layers={sorted(layers)}, weights={sorted(weights)}"
        )
    shapes = {arr.shape for arr in layers.values()}
    if len(shapes) != 1:
        raise ValueError(f"All layers must share one shape, got {shapes}.")

    score = np.zeros(next(iter(shapes)), dtype=float)
    for name, arr in layers.items():
        score += weights[name] * np.asarray(arr, dtype=float)
    return score
