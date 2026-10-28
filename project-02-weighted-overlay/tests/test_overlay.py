"""Tests for normalisation, weighting, ranking and the synthetic inputs."""

import numpy as np
import pytest

from site_overlay import (
    jaccard,
    make_grid,
    min_max,
    proximity_score,
    top_k_cells,
    validate_weights,
    weight_sweep,
    weighted_overlay,
)
from site_overlay.grid import (
    CELL_SIZE_M,
    distance_to_route_m,
    synthetic_exclusion_mask,
    synthetic_fibre_route,
    synthetic_groundwater,
)


def test_min_max_scales_to_unit_interval_and_keeps_nan():
    arr = np.array([[0.0, 5.0], [10.0, np.nan]])
    out = min_max(arr)
    assert np.isnan(out[1, 1])
    assert out[0, 0] == 0.0 and out[1, 0] == 1.0
    assert np.nanmin(out) >= 0 and np.nanmax(out) <= 1


def test_min_max_constant_layer_maps_to_zero():
    out = min_max(np.full((3, 3), 4.2))
    assert np.all(out == 0.0)


def test_min_max_rejects_all_nan():
    with pytest.raises(ValueError):
        min_max(np.full((2, 2), np.nan))


def test_proximity_score_rewards_shorter_distance():
    dist = np.array([0.0, 1000.0, 5000.0])
    score = proximity_score(dist)
    assert score[0] == 1.0 and score[-1] == 0.0
    assert np.all(np.diff(score) < 0)


@pytest.mark.parametrize(
    "weights",
    [{"a": 0.5, "b": 0.6}, {"a": -0.1, "b": 1.1}, {}],
)
def test_validate_weights_rejects_invalid_sets(weights):
    with pytest.raises(ValueError):
        validate_weights(weights)


def test_weighted_overlay_matches_hand_calculation():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    out = weighted_overlay({"a": a, "b": b}, {"a": 0.85, "b": 0.15})
    np.testing.assert_allclose(out, [[0.85, 0.15]])


def test_weighted_overlay_propagates_nan():
    a = np.array([[1.0, np.nan]])
    b = np.array([[0.5, 0.5]])
    out = weighted_overlay({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    assert np.isnan(out[0, 1]) and np.isfinite(out[0, 0])


def test_weighted_overlay_rejects_mismatched_names_and_shapes():
    a = np.zeros((2, 2))
    with pytest.raises(ValueError, match="names differ"):
        weighted_overlay({"a": a}, {"b": 1.0})
    with pytest.raises(ValueError, match="share one shape"):
        weighted_overlay({"a": a, "b": np.zeros((3, 3))}, {"a": 0.5, "b": 0.5})


def test_top_k_is_sorted_and_excludes_nan():
    score = np.array([[0.1, np.nan], [0.9, 0.5]])
    grid = make_grid((2, 2))
    top = top_k_cells(score, grid, k=10)
    assert len(top) == 3  # NaN cell dropped, fewer than k available
    assert list(top["score"]) == sorted(top["score"], reverse=True)
    assert top.iloc[0]["score"] == pytest.approx(0.9)


def test_top_k_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        top_k_cells(np.zeros((2, 2)), make_grid((2, 2)), k=0)


def test_jaccard_bounds():
    assert jaccard({1, 2}, {1, 2}) == 1.0
    assert jaccard({1}, {2}) == 0.0
    assert jaccard(set(), set()) == 1.0


def test_distance_to_route_is_zero_on_route_and_positive_elsewhere():
    grid = make_grid((20, 20))
    route = synthetic_fibre_route()
    dist = distance_to_route_m(grid, route)
    assert dist.shape == grid.shape
    assert np.all(dist >= 0)
    assert dist.min() < CELL_SIZE_M  # at least one cell lies near the route
    assert dist.max() > 0


def test_exclusion_mask_fraction_is_plausible_and_seeded():
    grid = make_grid((50, 50))
    mask_a = synthetic_exclusion_mask(grid, seed=11)
    mask_b = synthetic_exclusion_mask(grid, seed=11)
    np.testing.assert_array_equal(mask_a, mask_b)
    frac = mask_a.mean()
    assert 0.02 < frac < 0.3


def test_groundwater_is_nonnegative_and_seeded():
    grid = make_grid((30, 30))
    gw = synthetic_groundwater(grid, seed=7)
    assert gw.min() == pytest.approx(0.0)
    np.testing.assert_array_equal(gw, synthetic_groundwater(grid, seed=7))


def test_weight_sweep_base_case_is_identical_to_itself():
    rng = np.random.default_rng(0)
    layers = {"gw": rng.uniform(size=(20, 20)), "fibre": rng.uniform(size=(20, 20))}
    base = {"gw": 0.85, "fibre": 0.15}
    df = weight_sweep(layers, base, vary="gw", values=[0.85], k=20)
    assert df.loc[0, "spearman_vs_base"] == pytest.approx(1.0)
    assert df.loc[0, "top_k_jaccard_vs_base"] == pytest.approx(1.0)


def test_weight_sweep_returns_one_row_per_value_and_monotone_drift():
    rng = np.random.default_rng(5)
    layers = {"gw": rng.uniform(size=(30, 30)), "fibre": rng.uniform(size=(30, 30))}
    base = {"gw": 0.85, "fibre": 0.15}
    values = [0.5, 0.7, 0.85, 1.0]
    df = weight_sweep(layers, base, vary="gw", values=values, k=20)
    assert list(df["gw"]) == values
    # Moving further from the base weight should not increase rank agreement.
    rho = df.set_index("gw")["spearman_vs_base"]
    assert rho[0.5] <= rho[0.7] <= rho[0.85] + 1e-12
