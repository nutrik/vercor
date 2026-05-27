from pathlib import Path
from typing import Optional

import jax
from jax.typing import ArrayLike

from vercor.components import DataComponent
from vercor.forcing_data import read_forcing as _read_forcing
from vercor.grid import RectilinearGrid
from vercor.setups.data.assets import get_forcing_data
from vercor.setups.data._component_helpers import time_interpolated_data_component
from vercor.setups.data._field_helpers import (
    mask_time_last_surface_field,
    positive_binary_mask,
)

_ERA5_OCEAN_FIELD_NAMES = ("sea_surface_temperature",)


def _ocean_binary_mask_from_land_fraction(land_fraction: ArrayLike) -> jax.Array:
    """Convert a fractional land mask into a binary ocean mask."""

    return 1.0 - positive_binary_mask(land_fraction)


def make_era5_ocean(
    name: str = "OCN",
    surface_file: Optional[Path] = None,
) -> DataComponent:
    """Return an ERA5 ocean forcing component."""

    if surface_file is None:
        surface_file = get_forcing_data("era5_surface")

    data_files = {
        "surface": str(surface_file),
    }

    longitude = _read_forcing(data_files, "longitude", where="surface")
    latitude = _read_forcing(data_files, "latitude", where="surface")[::-1]
    land_fraction = _read_forcing(data_files, "lsm", where="surface", flip_y=True).T[
        0, ::
    ]
    binary_mask = _ocean_binary_mask_from_land_fraction(land_fraction)

    grid = RectilinearGrid(
        name=f"{name.lower()}-grid",
        longitude=longitude,
        latitude=latitude,
        binary_mask=binary_mask,
    )

    sea_surface_temperature = mask_time_last_surface_field(
        _read_forcing(data_files, "sst", where="surface", flip_y=True),
        binary_mask,
    )
    component = time_interpolated_data_component(
        name=name,
        grid=grid,
        fields={"sea_surface_temperature": sea_surface_temperature},
        outputs=_ERA5_OCEAN_FIELD_NAMES,
        data_files=data_files,
    )

    return component
