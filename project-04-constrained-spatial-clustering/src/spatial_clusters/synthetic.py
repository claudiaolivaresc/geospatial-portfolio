"""Synthetic study area and facility locations for the clustering demo.

Everything here is invented: the irregular boundary, the hotspot centres and
the densities. Nothing is derived from real facilities or real boundaries.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon

CRS = "EPSG:32633"  # WGS 84 / UTM zone 33N, metres
DEFAULT_CENTRE = (500_000.0, 5_400_000.0)


def study_area(radius_m: float = 20_000.0, centre: tuple[float, float] = DEFAULT_CENTRE) -> Polygon:
    """Irregular, valid polygon in metres built from a perturbed circle."""
    theta = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    r = radius_m * (1.0 + 0.2 * np.sin(3 * theta) + 0.1 * np.cos(5 * theta))
    coords = np.column_stack([centre[0] + r * np.cos(theta), centre[1] + r * np.sin(theta)])
    return Polygon(coords)


def facility_points(
    area: Polygon,
    n_points: int = 600,
    seed: int = 0,
    weights: tuple[float, ...] = (0.30, 0.25, 0.18, 0.12, 0.09, 0.06),
    spread_m: float = 2_500.0,
) -> np.ndarray:
    """Draw points from unequal hotspots, keeping only those inside the area.

    Unequal hotspot weights are deliberate. They create density differences
    that an unconstrained k-means will partly follow, which is what the
    capacity constraint is meant to correct.

    Returns:
        (n_points, 2) array of x, y coordinates in metres.
    """
    if n_points < 1:
        raise ValueError("n_points must be at least 1.")
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = area.bounds

    # Hotspot centres: sample inside the area, then keep the first ones found.
    hotspots: list[np.ndarray] = []
    while len(hotspots) < len(weights):
        cand = rng.uniform([minx, miny], [maxx, maxy], size=(200, 2))
        inside = cand[shapely.contains_xy(area, cand[:, 0], cand[:, 1])]
        hotspots.extend(list(inside))
    hotspots_arr = np.asarray(hotspots[: len(weights)])

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    points: list[np.ndarray] = []
    collected = 0
    while collected < n_points:
        batch = max(2 * (n_points - collected), 64)
        which = rng.choice(len(weights), size=batch, p=w)
        cand = hotspots_arr[which] + rng.normal(0.0, spread_m, size=(batch, 2))
        keep = cand[shapely.contains_xy(area, cand[:, 0], cand[:, 1])]
        points.append(keep)
        collected += len(keep)
    return np.vstack(points)[:n_points]
