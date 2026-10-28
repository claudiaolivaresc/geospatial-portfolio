"""Synthetic study-area grid and input layers for the site ranking demo.

Every layer is invented for demonstration. The fibre route, groundwater field
and exclusion area are generated with a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from rasterio.transform import Affine, from_origin
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString

CRS = "EPSG:32633"  # WGS 84 / UTM zone 33N, metres
CELL_SIZE_M = 250.0
ORIGIN_X, ORIGIN_Y = 600_000.0, 4_900_000.0  # upper-left corner


@dataclass(frozen=True)
class Grid:
    """Regular grid with an affine transform and CRS."""

    shape: tuple[int, int]
    transform: Affine
    crs: str = CRS

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return x and y coordinates (map units, metres) of every cell centre."""
        rows, cols = np.indices(self.shape)
        t = self.transform
        col_c, row_c = cols + 0.5, rows + 0.5  # cell centres, not corners
        xs = t.a * col_c + t.b * row_c + t.c
        ys = t.d * col_c + t.e * row_c + t.f
        return xs.astype(float), ys.astype(float)


def make_grid(shape: tuple[int, int] = (50, 50)) -> Grid:
    """Create the demonstration grid."""
    return Grid(shape=shape, transform=from_origin(ORIGIN_X, ORIGIN_Y, CELL_SIZE_M, CELL_SIZE_M))


def synthetic_groundwater(grid: Grid, seed: int = 7, smoothing_cells: float = 3.0) -> np.ndarray:
    """Smooth random groundwater availability index in relative units (0 to 1 after scaling).

    Smoothing is applied before any masking, so exclusion areas never leak
    NaN into the filter.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, grid.shape)
    smooth = gaussian_filter(raw, sigma=smoothing_cells, mode="nearest")
    return smooth - smooth.min()  # non-negative relative index


def synthetic_fibre_route() -> LineString:
    """A meandering fibre route across the study area, in map coordinates (metres)."""
    x0 = ORIGIN_X
    y_top = ORIGIN_Y
    width = CELL_SIZE_M * 50
    coords = [
        (x0 + 0.00 * width, y_top - 0.20 * width),
        (x0 + 0.30 * width, y_top - 0.35 * width),
        (x0 + 0.55 * width, y_top - 0.60 * width),
        (x0 + 0.80 * width, y_top - 0.55 * width),
        (x0 + 1.00 * width, y_top - 0.80 * width),
    ]
    return LineString(coords)


def distance_to_route_m(grid: Grid, route: LineString) -> np.ndarray:
    """Distance in metres from each cell centre to the route."""
    xs, ys = grid.cell_centres()
    points = shapely.points(xs, ys)  # keeps the 2-D shape of xs
    return shapely.distance(points, route)


def synthetic_exclusion_mask(grid: Grid, seed: int = 11, radius_cells: float = 5.0) -> np.ndarray:
    """Boolean mask, True where a cell is excluded (for example, a protected area).

    Two disks at random centres. Roughly 5 to 15% of the grid is excluded.
    """
    rng = np.random.default_rng(seed)
    rows, cols = grid.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    mask = np.zeros(grid.shape, dtype=bool)
    for _ in range(2):
        cy, cx = rng.uniform(0, rows), rng.uniform(0, cols)
        mask |= (rr - cy) ** 2 + (cc - cx) ** 2 <= radius_cells**2
    return mask
