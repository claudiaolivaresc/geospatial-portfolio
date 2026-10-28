"""Synthetic multi-band vegetation scene for the clustering demo.

The band names are generic placeholders for three passive-microwave
frequencies. The class profiles, noise levels and gaps are invented. The
scene is designed so that spectral shape and amplitude both carry signal.
Nothing here is an observation or a validated VOD product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

BAND_NAMES = ("low_freq", "mid_freq", "high_freq")
CLASS_NAMES = ("forest", "shrubland", "cropland", "grassland")

# Mean VOD per band for each class (invented, not measured).
CLASS_MEANS = np.array(
    [
        [0.95, 1.05, 1.15],  # forest
        [0.45, 0.50, 0.55],  # shrubland
        [0.30, 0.35, 0.28],  # cropland: flatter, high band lower
        [0.20, 0.22, 0.25],  # grassland
    ]
)
CLASS_SD = np.array([0.08, 0.06, 0.05, 0.04])
MOISTURE_EFFECT = 0.15  # common amplitude factor shared by all bands in a pixel
GAP_FRACTION = 0.03  # random missing observations per band


@dataclass(frozen=True)
class Scene:
    """Multi-band VOD array, land-cover reference and band/class names."""

    bands: np.ndarray  # (n_bands, rows, cols), NaN where missing
    land_cover: np.ndarray  # (rows, cols) integer class index
    band_names: tuple[str, ...] = BAND_NAMES
    class_names: tuple[str, ...] = CLASS_NAMES


def make_land_cover(shape: tuple[int, int], rng: np.random.Generator, sigma: float = 6.0) -> np.ndarray:
    """Smooth random field quantised into four classes of unequal, blobby extent."""
    field = gaussian_filter(rng.normal(size=shape), sigma=sigma, mode="nearest")
    edges = np.quantile(field, [0.35, 0.65, 0.85])
    return np.digitize(field, edges).astype(np.int64)


def make_scene(shape: tuple[int, int] = (120, 120), seed: int = 0) -> Scene:
    """Generate a seeded synthetic scene."""
    rng = np.random.default_rng(seed)
    land_cover = make_land_cover(shape, rng)
    moisture = gaussian_filter(rng.normal(size=shape), sigma=4.0, mode="nearest")
    moisture = moisture / moisture.std()

    bands = np.empty((len(BAND_NAMES), *shape), dtype=np.float32)
    for b in range(len(BAND_NAMES)):
        mean = CLASS_MEANS[land_cover, b] * (1.0 + MOISTURE_EFFECT * moisture)
        noise = rng.normal(0.0, 1.0, shape) * CLASS_SD[land_cover]
        bands[b] = mean + noise

    gaps = rng.uniform(size=bands.shape) < GAP_FRACTION
    bands[gaps] = np.nan
    return Scene(bands=bands, land_cover=land_cover)
