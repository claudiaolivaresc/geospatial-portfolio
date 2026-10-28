"""Synthetic input rasters for the Monte Carlo demo.

All values are invented for demonstration. They are not geothermal data and
must not be read as a real resource estimate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:32633"  # WGS 84 / UTM zone 33N, units in metres
CELL_SIZE_M = 500.0
ORIGIN_X, ORIGIN_Y = 500_000.0, 5_000_000.0

# Parameter name -> file stem. Each parameter has a mean and a standard deviation raster.
PARAMETERS = ("thickness_m", "porosity", "delta_t_k")


def make_parameter_surfaces(
    shape: tuple[int, int] = (25, 25), seed: int = 42
) -> dict[str, np.ndarray]:
    """Return mean and sd surfaces for each parameter, keyed as ``<name>_mean``/``<name>_sd``."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    trend = (rr + cc) / (rows + cols - 2)  # smooth gradient in [0, 1]
    smooth_noise = rng.normal(0.0, 1.0, shape)

    thickness_mean = 800.0 + 600.0 * trend + 40.0 * smooth_noise
    porosity_mean = np.clip(0.08 + 0.02 * smooth_noise - 0.02 * trend, 0.02, 0.3)
    delta_t_mean = 120.0 + 30.0 * trend + 5.0 * smooth_noise

    return {
        "thickness_m_mean": thickness_mean,
        "thickness_m_sd": 0.10 * thickness_mean,
        "porosity_mean": porosity_mean,
        "porosity_sd": 0.01 * np.ones(shape),
        "delta_t_k_mean": delta_t_mean,
        "delta_t_k_sd": 0.08 * delta_t_mean,
    }


def write_input_rasters(out_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    """Write the synthetic surfaces as aligned GeoTIFFs and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surfaces = make_parameter_surfaces(seed=seed)
    transform = from_origin(ORIGIN_X, ORIGIN_Y, CELL_SIZE_M, CELL_SIZE_M)
    height, width = next(iter(surfaces.values())).shape

    paths: dict[str, Path] = {}
    for name, data in surfaces.items():
        path = out_dir / f"{name}.tif"
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": width,
            "height": height,
            "crs": CRS,
            "transform": transform,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)
        paths[name] = path
    return paths
