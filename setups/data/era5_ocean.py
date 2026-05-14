from pathlib import Path
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.components.base import DataComponent, data_component
from vercor.dtypes import as_jax_real_array
from vercor.field_layout import canonicalize_time_last_surface_field
from vercor.grid import RectilinearGrid
from vercor.assets import get_forcing_data
from setups.data.forcing import read_forcing as _read_forcing

_ERA5_OCEAN_FIELD_NAMES = ("sea_surface_temperature",)


def _ocean_binary_mask_from_land_fraction(land_fraction: ArrayLike) -> jax.Array:
    """Convert a fractional land mask into a binary ocean mask."""
    land_fraction_array = as_jax_real_array(land_fraction)
    return 1.0 - jnp.where(land_fraction_array > 0.0, 1.0, 0.0)


def _mask_sea_surface_temperature(
    sea_surface_temperature: ArrayLike,
    binary_mask: ArrayLike,
) -> jax.Array:
    """Apply the binary ocean mask and return a `(nTime, nLat, nLon)` SST field."""
    return (
        canonicalize_time_last_surface_field(sea_surface_temperature)
        * jnp.where(
            as_jax_real_array(binary_mask) > 0.0,
            1.0,
            jnp.nan,
        )[jnp.newaxis, ...]
    )


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

    sea_surface_temperature = _mask_sea_surface_temperature(
        _read_forcing(data_files, "sst", where="surface", flip_y=True),
        binary_mask,
    )
    component = data_component(
        name=name,
        grid=grid,
        fields={"sea_surface_temperature": sea_surface_temperature},
    )
    component.declare_fields(outputs=_ERA5_OCEAN_FIELD_NAMES)
    component.update_settings(apply_time_interpolation=True)
    cast(Any, component).DATA_FILES = data_files

    return component
