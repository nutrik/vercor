from pathlib import Path
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.components.base import DataComponent, data_component
from vercor.dtypes import as_jax_real_array, jax_arange
from vercor.field_layout import canonicalize_time_last_surface_field
from vercor.grid import RectilinearGrid
from vercor.assets import get_forcing_data
from setups.data.forcing import read_forcing as _read_forcing

_ERAINTERIM_OCEAN_FIELD_NAMES = ("sea_surface_temperature",)


def _assemble_erainterim_latitude(
    latitude_core: ArrayLike,
    full_latitude: ArrayLike,
    latitude_start: int,
    latitude_stop: int,
) -> jax.Array:
    """Insert the ERA-Interim latitude band into the full global latitude vector."""
    return (
        as_jax_real_array(full_latitude)
        .at[latitude_start:latitude_stop]
        .set(as_jax_real_array(latitude_core))
    )


def _assemble_erainterim_field(
    core_field: ArrayLike,
    full_latitude_size: int,
    latitude_start: int,
    latitude_stop: int,
    longitude_roll: int = 0,
    offset: float = 0.0,
) -> jax.Array:
    """Embed a `(nlon, nlat_core, time)` field into the full global ocean grid."""
    core_field_array = as_jax_real_array(core_field) + offset
    full_field = jnp.zeros(
        (
            int(core_field_array.shape[0]),
            full_latitude_size,
            int(core_field_array.shape[2]),
        ),
        dtype=core_field_array.dtype,
    )
    full_field = full_field.at[:, latitude_start:latitude_stop, :].set(core_field_array)
    if longitude_roll != 0:
        return jnp.roll(full_field, longitude_roll, axis=0)
    return full_field


def _binary_ocean_mask_from_salinity(salinity: ArrayLike) -> jax.Array:
    """Create a binary ocean mask from a full-grid salinity field."""
    return jnp.where(as_jax_real_array(salinity) > 0.0, 1.0, 0.0)[..., 0].T


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


def make_erainterim_ocean(
    name: str = "OCN",
    resolution: str = "4deg",
    model_level_file: Optional[Path] = None,
) -> DataComponent:
    """Return an ERA-Interim ocean forcing component."""

    if model_level_file is None:
        model_level_file = get_forcing_data(f"erainterim_ocean_{resolution}")

    data_files = {
        "model_level": str(model_level_file),
    }

    longitude = _read_forcing(data_files, "xt", where="model_level")
    grid_step = float(longitude[1] - longitude[0])
    yt_bndry = 89.5 if grid_step == 1 else 90.0
    latitude_start = 10 if grid_step == 1 else 3
    full_latitude = jax_arange(-yt_bndry, yt_bndry + grid_step, grid_step)
    latitude_stop = int(full_latitude.size) - latitude_start
    longitude_roll = 90 if grid_step == 1 else 0

    # To cover the whole globe with 1 or 4 degree resolution
    latitude = _assemble_erainterim_latitude(
        _read_forcing(data_files, "yt", where="model_level"),
        full_latitude,
        latitude_start,
        latitude_stop,
    )
    longitude = longitude - 90.0 if grid_step == 1 else longitude

    sss = _assemble_erainterim_field(
        _read_forcing(data_files, "sss", where="model_level"),
        int(full_latitude.size),
        latitude_start,
        latitude_stop,
        longitude_roll=longitude_roll,
    )
    binary_mask = _binary_ocean_mask_from_salinity(sss)

    grid = RectilinearGrid(
        name=f"{name.lower()}-grid",
        longitude=longitude,
        latitude=latitude,
        binary_mask=binary_mask,
    )

    sst = _mask_sea_surface_temperature(
        _assemble_erainterim_field(
            _read_forcing(data_files, "sst", where="model_level"),
            int(full_latitude.size),
            latitude_start,
            latitude_stop,
            longitude_roll=longitude_roll,
            offset=273.15,
        ),
        binary_mask,
    )

    component = data_component(
        name=name,
        grid=grid,
        fields={"sea_surface_temperature": sst},
    )
    component.declare_fields(outputs=_ERAINTERIM_OCEAN_FIELD_NAMES)
    component.update_settings(apply_time_interpolation=True)
    cast(Any, component).DATA_FILES = data_files
    return component
