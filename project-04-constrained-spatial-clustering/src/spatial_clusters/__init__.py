"""Capacity-constrained spatial clustering with a network cost estimate (synthetic demo)."""

from spatial_clusters.clustering import (
    ClusterResult,
    assign_with_capacity,
    capacitated_kmeans,
    capacity_for,
    compare_methods,
    diameter_km,
    mst_length_km,
    unconstrained_kmeans,
)
from spatial_clusters.geometry import validate_points, validate_study_area
from spatial_clusters.synthetic import CRS, facility_points, study_area

__all__ = [
    "CRS",
    "ClusterResult",
    "assign_with_capacity",
    "capacitated_kmeans",
    "capacity_for",
    "compare_methods",
    "diameter_km",
    "facility_points",
    "mst_length_km",
    "study_area",
    "unconstrained_kmeans",
    "validate_points",
    "validate_study_area",
]
