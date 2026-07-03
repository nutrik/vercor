"""Public grid constructors and grid types."""

from typing import Any

from vercor.grid import RectilinearGrid
from vercor.grid_geometry import make_rectilinear_grid


def rectilinear(
    name: str,
    nlon: int,
    nlat: int,
    longitude_start: float,
    longitude_end: float,
    latitude_start: float,
    latitude_end: float,
    mask: Any | None = None,
) -> RectilinearGrid:
    """Build a rectilinear grid with equally spaced coordinate centers."""

    return make_rectilinear_grid(
        name,
        nlon,
        nlat,
        longitude_start,
        longitude_end,
        latitude_start,
        latitude_end,
        mask=mask,
    )


__all__ = ["RectilinearGrid", "rectilinear"]
