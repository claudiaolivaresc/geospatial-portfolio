"""Input validation for the study area and point locations."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.validation import explain_validity


def validate_study_area(area: Polygon) -> Polygon:
    """Return the polygon if it is a valid, non-empty areal geometry, else raise ValueError."""
    if not isinstance(area, Polygon):
        raise ValueError(f"Study area must be a Polygon, got {type(area).__name__}.")
    if area.is_empty:
        raise ValueError("Study area is empty.")
    if not area.is_valid:
        raise ValueError(f"Study area is invalid: {explain_validity(area)}")
    if area.area <= 0:
        raise ValueError("Study area has zero area.")
    return area


def validate_points(points: np.ndarray, area: Polygon) -> np.ndarray:
    """Check shape, finiteness, duplicates and containment. Returns the points unchanged."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Points must have shape (n, 2), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Points contain NaN or infinite coordinates.")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Points contain duplicate locations.")
    outside = ~shapely.contains_xy(area, points[:, 0], points[:, 1])
    if outside.any():
        raise ValueError(f"{int(outside.sum())} point(s) lie outside the study area.")
    return points
