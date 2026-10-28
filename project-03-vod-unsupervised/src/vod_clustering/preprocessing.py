"""Harmonisation and spatial splitting.

Harmonisation here means: keep finite values inside a physical range, reshape
to one row per pixel, and work in log space. A common multiplicative moisture
effect becomes an additive offset in log space, so the band-to-band pattern
is easier to compare. Standardisation is left to the modelling pipeline, so
that it is fitted on training data only.
"""

from __future__ import annotations

import numpy as np

VALID_RANGE = (1e-3, 2.5)  # VOD outside this range is treated as invalid


def harmonise(
    bands: np.ndarray, valid_range: tuple[float, float] = VALID_RANGE
) -> tuple[np.ndarray, np.ndarray]:
    """Return log-VOD features for valid pixels and the 2-D valid-pixel mask.

    Args:
        bands: array of shape (n_bands, rows, cols). NaN marks missing values.
        valid_range: inclusive (low, high) range of usable values.

    Returns:
        features: (n_valid, n_bands) array of log(VOD).
        valid: (rows, cols) boolean mask. A pixel is valid only if every band is valid.
    """
    if bands.ndim != 3:
        raise ValueError(f"bands must be 3-D (band, row, col), got {bands.ndim}-D.")
    low, high = valid_range
    if not low < high:
        raise ValueError("valid_range must be (low, high) with low < high.")

    in_range = np.isfinite(bands) & (bands >= low) & (bands <= high)
    valid = in_range.all(axis=0)
    values = np.moveaxis(bands, 0, -1)[valid]  # (n_valid, n_bands)
    return np.log(values.astype(float)), valid


def spatial_split(
    shape: tuple[int, int], train_frac: float = 0.5, buffer_cols: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Left-right split by column with a buffer, to limit spatial leakage.

    Neighbouring pixels are strongly correlated, so pixels right next to the
    boundary would leak information between train and test. The buffer columns
    belong to neither set.

    Returns:
        train_mask, test_mask: boolean (rows, cols) arrays, disjoint by construction.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    rows, cols = shape
    split = int(round(cols * train_frac))
    if split - buffer_cols < 1 or split + buffer_cols >= cols:
        raise ValueError("Buffer too wide for this grid and split fraction.")
    col_idx = np.arange(cols)
    train_cols = col_idx < split - buffer_cols
    test_cols = col_idx >= split + buffer_cols
    train_mask = np.broadcast_to(train_cols, shape).copy()
    test_mask = np.broadcast_to(test_cols, shape).copy()
    return train_mask, test_mask
