"""Grid consistency checks for raster inputs."""

from __future__ import annotations

from collections.abc import Sequence

import rasterio


def assert_aligned(paths: Sequence[str]) -> rasterio.io.DatasetReader | None:
    """Raise ValueError unless all rasters share CRS, transform and shape.

    Only metadata is read; no pixel data is loaded.
    """
    if not paths:
        raise ValueError("At least one raster path is required.")

    reference = None
    for path in paths:
        with rasterio.open(path) as src:
            meta = (src.crs, src.transform, src.width, src.height)
        if reference is None:
            reference = (path, meta)
            continue
        ref_path, ref_meta = reference
        if meta != ref_meta:
            raise ValueError(
                f"Grid mismatch between {ref_path} and {path}: "
                f"reference={ref_meta}, found={meta}"
            )
    return None
