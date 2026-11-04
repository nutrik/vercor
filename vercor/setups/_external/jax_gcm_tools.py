from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from importlib import resources
from pathlib import Path
from typing import Optional, Tuple, cast

import xarray as xr

from dinosaur.coordinate_systems import CoordinateSystem

from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.params import Parameters

from vercor.dtypes import as_jax_real_array


def _prefer_h5netcdf() -> AbstractContextManager[object]:
    """Temporarily prefer h5netcdf when xarray supports engine ordering."""

    options = xr.get_options()
    if "netcdf_engine_order" not in options:
        return nullcontext()
    configured = tuple(cast(Sequence[str], options["netcdf_engine_order"]))
    preferred = (
        "h5netcdf",
        *(engine for engine in configured if engine != "h5netcdf"),
    )
    return xr.set_options(netcdf_engine_order=preferred)


def change_jcm_parameter_values(
    parameters: Mapping[str, float], default_parameters: Parameters
) -> None:

    for key, value in parameters.items():
        parameter_group_name, parameter_name = key.split(".")
        default_parameters.__getattribute__(parameter_group_name).__setattr__(
            parameter_name, as_jax_real_array(value)
        )


def load_jcm_coords_terrain_forcing(
    resolution: int = 31,
    input_data_directory: Optional[Path] = None,
) -> Tuple[CoordinateSystem, TerrainData, ForcingData]:
    """Generate JCM coordinates, forcing and topography files at the specified resolution.

    Arguments:
        resolution: Optional resolution of the JCM files to generate (e.g., 31 for T31)
        input_data_directory: Optional directory to look for existing files and to save generated files.
                              If None, defaults to ~/.vercor/jcm/

    Returns:
        A tuple of (coords, terrain, forcing) objects.
    """

    coords = get_speedy_coords(
        spectral_truncation=resolution
    )  # T31 spectral resolution with 8 vertical levels

    # Read JCM topography file
    # Load realistic orography and land-sea mask, interpolated to T31 grid
    if input_data_directory is None:
        data_dir = resources.files("jcm.data.bc.t30.clim")
    else:
        data_dir = Path(input_data_directory)

    terrain_file = data_dir / "terrain.nc"

    # Load realistic forcing data (SST, sea ice, soil moisture, etc.) interpolated to T31 grid
    forcing_file = data_dir / "forcing.nc"
    with _prefer_h5netcdf():
        terrain = TerrainData.from_file(terrain_file, coords=coords)
        forcing = ForcingData.from_file(forcing_file, coords=coords)

    return (coords, terrain, forcing)
