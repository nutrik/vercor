from pathlib import Path
from typing import Optional

import jax
from jax.typing import ArrayLike

from vercor.components.base import DataComponent
from vercor.dtypes import as_jax_real_array
from vercor.field_layout import canonicalize_time_last_surface_field
from vercor.grid import RectilinearGrid
from vercor.assets import get_forcing_data
from setups.data._component_helpers import time_interpolated_data_component
from setups.data.forcing import read_forcing as _read_forcing

_ERA5_LAND_FIELD_NAMES = ("land_surface_temperature",)


def _prepare_era5_land_runtime_fields(
    longitude: ArrayLike,
    latitude: ArrayLike,
    binary_mask: ArrayLike,
    land_surface_temperature: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Normalize ERA5 land forcing arrays for JAX-backed runtime storage."""
    return (
        as_jax_real_array(longitude),
        as_jax_real_array(latitude),
        as_jax_real_array(binary_mask).T,
        canonicalize_time_last_surface_field(land_surface_temperature),
    )


def make_era5_land(
    name: str = "LND",
    surface_file: Optional[Path] = None,
) -> DataComponent:
    """Return an ERA5 land forcing component."""

    if surface_file is None:
        surface_file = get_forcing_data("era5_land_masked")

    data_files = {
        "surface": str(surface_file),
    }

    (
        longitude,
        latitude,
        binary_mask,
        land_surface_temperature,
    ) = _prepare_era5_land_runtime_fields(
        _read_forcing(data_files, "lon", where="surface"),
        _read_forcing(data_files, "lat", where="surface"),
        _read_forcing(data_files, "mask", where="surface"),
        _read_forcing(data_files, "skt", where="surface"),
    )
    grid = RectilinearGrid(
        name=f"{name.lower()}-grid",
        longitude=longitude,
        latitude=latitude,
        binary_mask=binary_mask,
    )

    component = time_interpolated_data_component(
        name=name,
        grid=grid,
        fields={"land_surface_temperature": land_surface_temperature},
        outputs=_ERA5_LAND_FIELD_NAMES,
        data_files=data_files,
    )

    return component
