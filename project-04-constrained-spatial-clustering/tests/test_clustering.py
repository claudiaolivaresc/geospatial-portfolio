"""Tests for capacity constraints, network metrics, validation and synthetic inputs."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from spatial_clusters import (
    assign_with_capacity,
    capacitated_kmeans,
    capacity_for,
    diameter_km,
    facility_points,
    mst_length_km,
    study_area,
    unconstrained_kmeans,
    validate_points,
    validate_study_area,
)


def test_capacity_formula_adds_slack_and_rounds_up():
    assert capacity_for(100, 4, slack=0.0) == 25
    assert capacity_for(100, 4, slack=0.1) == 28  # ceil(27.5)
    with pytest.raises(ValueError):
        capacity_for(3, 5)


def test_assignment_respects_capacity_and_assigns_every_point():
    rng = np.random.default_rng(0)
    points = rng.uniform(size=(30, 2))
    centres = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    labels = assign_with_capacity(points, centres, capacity=10)
    counts = np.bincount(labels, minlength=3)
    assert counts.sum() == 30
    assert counts.max() <= 10


def test_assignment_rejects_infeasible_capacity():
    points = np.zeros((10, 2)) + np.arange(10)[:, None]
    with pytest.raises(ValueError, match="Infeasible"):
        assign_with_capacity(points, np.zeros((2, 2)), capacity=4)


def test_capacitated_kmeans_never_exceeds_capacity_and_is_deterministic():
    area = study_area()
    points = facility_points(area, n_points=300, seed=1)
    cap = capacity_for(len(points), 5, slack=0.1)
    a = capacitated_kmeans(points, 5, capacity=cap, seed=0)
    b = capacitated_kmeans(points, 5, capacity=cap, seed=0)
    assert np.bincount(a.labels).max() <= cap
    np.testing.assert_array_equal(a.labels, b.labels)


def test_capacity_constraint_binds_on_unequal_density():
    # Unequal hotspots make plain k-means exceed a tight equal-share capacity.
    area = study_area()
    points = facility_points(area, n_points=400, seed=2)
    k = 4
    cap = capacity_for(len(points), k, slack=0.05)
    plain = unconstrained_kmeans(points, k, seed=0)
    constrained = capacitated_kmeans(points, k, capacity=cap, seed=0)
    assert np.bincount(plain.labels).max() > cap
    assert np.bincount(constrained.labels).max() <= cap


def test_mst_length_of_collinear_points_is_span():
    # Points at 0, 1, 3 km on a line: MST links 0-1 and 1-3, total 3 km.
    pts = np.array([[0.0, 0.0], [1000.0, 0.0], [3000.0, 0.0]])
    assert mst_length_km(pts) == pytest.approx(3.0)


def test_mst_of_single_point_is_zero():
    assert mst_length_km(np.array([[5.0, 5.0]])) == 0.0


def test_diameter_of_square_is_its_diagonal():
    square = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0], [1000.0, 1000.0]])
    assert diameter_km(square) == pytest.approx(np.sqrt(2.0))


def test_validate_study_area_rejects_bowtie():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    with pytest.raises(ValueError, match="invalid"):
        validate_study_area(bowtie)


def test_validate_study_area_accepts_synthetic_area():
    assert validate_study_area(study_area()).is_valid


def test_validate_points_rejects_outside_and_duplicates():
    area = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    with pytest.raises(ValueError, match="outside"):
        validate_points(np.array([[5.0, 5.0], [20.0, 5.0]]), area)
    with pytest.raises(ValueError, match="duplicate"):
        validate_points(np.array([[5.0, 5.0], [5.0, 5.0]]), area)
    with pytest.raises(ValueError, match="NaN"):
        validate_points(np.array([[5.0, np.nan]]), area)


def test_facility_points_are_inside_area_and_seeded():
    area = study_area()
    a = facility_points(area, n_points=200, seed=3)
    b = facility_points(area, n_points=200, seed=3)
    np.testing.assert_array_equal(a, b)
    validate_points(a, area)  # raises if any point falls outside
    assert len(a) == 200
