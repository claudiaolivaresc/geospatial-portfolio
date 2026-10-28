"""Correlated Monte Carlo sampling and resource aggregation.

Resource proxy per cell (energy in place, illustrative only):

    E = A * H * phi * dT * rho_c

where A is cell area (m^2), H thickness (m), phi porosity (-), dT temperature
contrast (K) and rho_c volumetric heat capacity (J m^-3 K^-1). Results are in
petajoules (PJ = 1e15 J).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

RHO_C_J_PER_M3_K = 2.5e6  # assumed volumetric heat capacity of rock and water
PARAM_ORDER = ("thickness_m", "porosity", "delta_t_k")


def cholesky_factor(corr: np.ndarray) -> np.ndarray:
    """Return the lower Cholesky factor of a correlation matrix.

    Raises ValueError for non-square, asymmetric, non-unit-diagonal or
    non-positive-definite input.
    """
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"Correlation matrix must be square, got shape {corr.shape}.")
    if not np.allclose(corr, corr.T):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(corr), 1.0):
        raise ValueError("Correlation matrix must have a unit diagonal.")
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Correlation matrix is not positive definite.") from exc


def sample_parameters(
    means: np.ndarray,
    sds: np.ndarray,
    corr: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    spatial: str = "independent",
) -> np.ndarray:
    """Draw correlated parameter fields.

    Args:
        means: shape (n_cells, k) per-cell parameter means.
        sds: shape (n_cells, k) per-cell standard deviations.
        corr: (k, k) correlation matrix between parameters within a cell.
        n_draws: number of Monte Carlo iterations.
        rng: NumPy random generator (pass a seeded one for reproducibility).
        spatial: ``"independent"`` draws separate errors per cell, so cell
            errors average out when summed. ``"shared"`` applies one error
            draw to every cell, which is the fully spatially correlated case.

    Returns:
        Array of shape (n_draws, n_cells, k).
    """
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    if means.shape != sds.shape or means.ndim != 2:
        raise ValueError("means and sds must share shape (n_cells, k).")
    n_cells, k = means.shape
    if corr.shape != (k, k):
        raise ValueError(f"corr must have shape ({k}, {k}), got {corr.shape}.")
    if np.any(sds < 0):
        raise ValueError("Standard deviations must be non-negative.")
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1.")

    chol = cholesky_factor(corr)
    if spatial == "independent":
        z = rng.standard_normal((n_draws, n_cells, k))
    elif spatial == "shared":
        z = rng.standard_normal((n_draws, 1, k))
    else:
        raise ValueError("spatial must be 'independent' or 'shared'.")

    x = z @ chol.T  # correlated standard normals, broadcast over cells if shared
    return means[None, :, :] + sds[None, :, :] * x


def apply_physical_bounds(samples: np.ndarray) -> np.ndarray:
    """Clip sampled parameters to physical ranges (thickness, dT >= 0; 0 < porosity < 1)."""
    out = samples.copy()
    out[..., 0] = np.maximum(out[..., 0], 0.0)  # thickness_m
    out[..., 1] = np.clip(out[..., 1], 1e-3, 0.999)  # porosity
    out[..., 2] = np.maximum(out[..., 2], 0.0)  # delta_t_k
    return out


def resource_pj(
    thickness_m: np.ndarray,
    porosity: np.ndarray,
    delta_t_k: np.ndarray,
    cell_area_m2: float,
) -> np.ndarray:
    """Energy-in-place proxy per cell in PJ. Inputs broadcast together."""
    joules = cell_area_m2 * thickness_m * porosity * delta_t_k * RHO_C_J_PER_M3_K
    return joules / 1e15


@dataclass(frozen=True)
class MonteCarloResult:
    """Container for Monte Carlo output."""

    totals_pj: np.ndarray  # (n_draws,) regional totals
    samples: np.ndarray  # (n_draws, n_cells, k) bounded parameter samples
    spatial: str


def run_monte_carlo(
    means: np.ndarray,
    sds: np.ndarray,
    corr: np.ndarray,
    cell_area_m2: float,
    n_draws: int,
    seed: int,
    spatial: str = "independent",
    keep_samples: bool = True,
) -> MonteCarloResult:
    """Run the full sampling and aggregation pipeline for one spatial mode."""
    rng = np.random.default_rng(seed)
    samples = apply_physical_bounds(
        sample_parameters(means, sds, corr, n_draws, rng, spatial=spatial)
    )
    cells = resource_pj(samples[..., 0], samples[..., 1], samples[..., 2], cell_area_m2)
    totals = cells.sum(axis=1)
    return MonteCarloResult(
        totals_pj=totals,
        samples=samples if keep_samples else np.empty((0,)),
        spatial=spatial,
    )


def summarise_totals(totals: np.ndarray) -> dict[str, float]:
    """Summarise a distribution of totals with P90 / P50 / P10 (petroleum convention).

    P90 is the low case, exceeded with 90% probability, so it is the 10th
    percentile of the sample. P10 is the high case, the 90th percentile.
    """
    p90, p50, p10 = np.percentile(totals, [10, 50, 90])
    return {"P90": float(p90), "P50": float(p50), "P10": float(p10), "mean": float(totals.mean())}


def parameter_sensitivity(samples: np.ndarray, totals: np.ndarray) -> dict[str, float]:
    """Spearman rank correlation between each parameter's regional mean and the total.

    Averaging over cells removes per-cell noise, so the ranks reflect the
    regional driver of each parameter.
    """
    result = {}
    for j, name in enumerate(PARAM_ORDER):
        regional_mean = samples[:, :, j].mean(axis=1)
        rho, _ = stats.spearmanr(regional_mean, totals)
        result[name] = float(rho)
    return result
