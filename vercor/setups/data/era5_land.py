from pathlib import Path
from typing import Optional

from vercor.components import DataComponent
from vercor.dtypes import as_jax_real_array
from vercor.field_layout import canonicalize_time_last_surface_field
from vercor.forcing_data import read_forcing as _read_forcing
from vercor.grid import RectilinearGrid
from vercor.setups.data.assets import get_forcing_data
from vercor.setups.data._component_helpers import time_interpolated_data_component

_ERA5_LAND_FIELD_NAMES = ("land_surface_temperature",)


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

    longitude = as_jax_real_array(_read_forcing(data_files, "lon", where="surface"))
    latitude = as_jax_real_array(_read_forcing(data_files, "lat", where="surface"))
    binary_mask = as_jax_real_array(
        _read_forcing(data_files, "mask", where="surface")
    ).T
    land_surface_temperature = canonicalize_time_last_surface_field(
        _read_forcing(data_files, "skt", where="surface")
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
