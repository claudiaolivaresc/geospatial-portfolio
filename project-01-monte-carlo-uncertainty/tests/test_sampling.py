"""Tests for correlated sampling, aggregation and grid checks."""

import numpy as np
import pytest
from rasterio.transform import from_origin

from geothermal_mc import assert_aligned
from geothermal_mc.sampling import (
    apply_physical_bounds,
    cholesky_factor,
    resource_pj,
    run_monte_carlo,
    sample_parameters,
    summarise_totals,
)
from geothermal_mc.synthetic import write_input_rasters

CORR_3 = np.array(
    [
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 1.0],
    ]
)


def test_cholesky_reconstructs_correlation_matrix():
    chol = cholesky_factor(CORR_3)
    np.testing.assert_allclose(chol @ chol.T, CORR_3, atol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[1.0, 2.0], [2.0, 1.0]]),  # |rho| > 1, not positive definite
        np.array([[1.0, 0.1], [0.2, 1.0]]),  # asymmetric
        np.array([[2.0, 0.0], [0.0, 1.0]]),  # diagonal not unity
        np.ones((2, 3)),  # not square
    ],
)
def test_cholesky_rejects_invalid_matrices(bad):
    with pytest.raises(ValueError):
        cholesky_factor(bad)


def test_sampled_correlation_matches_target():
    n_cells = 200
    means = np.zeros((n_cells, 3))
    sds = np.ones((n_cells, 3))
    rng = np.random.default_rng(0)
    samples = sample_parameters(means, sds, CORR_3, n_draws=2000, rng=rng)
    flat = samples.reshape(-1, 3)
    empirical = np.corrcoef(flat, rowvar=False)
    np.testing.assert_allclose(empirical, CORR_3, atol=0.02)


def test_shared_mode_applies_one_draw_to_all_cells():
    means = np.tile([100.0, 0.1, 50.0], (30, 1))
    sds = np.tile([10.0, 0.01, 5.0], (30, 1))
    rng = np.random.default_rng(1)
    samples = sample_parameters(means, sds, CORR_3, n_draws=50, rng=rng, spatial="shared")
    # Every cell in a draw carries the same value when means and sds are uniform.
    assert np.allclose(samples, samples[:, :1, :])


def test_independent_mode_varies_across_cells():
    means = np.tile([100.0, 0.1, 50.0], (30, 1))
    sds = np.tile([10.0, 0.01, 5.0], (30, 1))
    rng = np.random.default_rng(1)
    samples = sample_parameters(means, sds, CORR_3, n_draws=50, rng=rng, spatial="independent")
    assert samples[:, :, 0].std(axis=1).mean() > 0


def test_monte_carlo_is_reproducible_with_seed():
    means = np.tile([800.0, 0.08, 120.0], (16, 1))
    sds = np.tile([80.0, 0.01, 10.0], (16, 1))
    a = run_monte_carlo(means, sds, CORR_3, 2.5e5, n_draws=100, seed=7)
    b = run_monte_carlo(means, sds, CORR_3, 2.5e5, n_draws=100, seed=7)
    np.testing.assert_array_equal(a.totals_pj, b.totals_pj)


def test_resource_units_for_one_cell():
    # 1 m^2 cell, 1 m thick, porosity 1, dT 1 K -> rho_c J = 2.5e6 J = 2.5e-9 PJ
    value = resource_pj(np.array(1.0), np.array(1.0), np.array(1.0), cell_area_m2=1.0)
    assert float(value) == pytest.approx(2.5e6 / 1e15)


def test_percentile_convention_orders_low_case_below_high_case():
    totals = np.random.default_rng(3).lognormal(mean=3.0, sigma=0.2, size=5000)
    summary = summarise_totals(totals)
    assert summary["P90"] < summary["P50"] < summary["P10"]


def test_physical_bounds_remove_negative_thickness_and_bad_porosity():
    raw = np.array([[[-5.0, 1.5, -2.0]]])
    bounded = apply_physical_bounds(raw)
    assert bounded[0, 0, 0] == 0.0
    assert 0.0 < bounded[0, 0, 1] < 1.0
    assert bounded[0, 0, 2] == 0.0


def test_invalid_spatial_mode_raises():
    with pytest.raises(ValueError):
        sample_parameters(
            np.ones((2, 3)), np.ones((2, 3)), CORR_3, 5, np.random.default_rng(0), spatial="bogus"
        )


def test_assert_aligned_accepts_matching_grids(tmp_path):
    paths = write_input_rasters(tmp_path)
    assert_aligned([str(p) for p in paths.values()])  # should not raise


def test_assert_aligned_detects_shifted_grid(tmp_path):
    import rasterio

    paths = write_input_rasters(tmp_path)
    ref = str(paths["porosity_mean"])
    shifted = tmp_path / "shifted.tif"
    with rasterio.open(ref) as src:
        profile = src.profile.copy()
        data = src.read()
    profile["transform"] = from_origin(500_000.0 + 500.0, 5_000_000.0, 500.0, 500.0)
    with rasterio.open(shifted, "w", **profile) as dst:
        dst.write(data)
    with pytest.raises(ValueError, match="Grid mismatch"):
        assert_aligned([ref, str(shifted)])
