import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dinosaur.coordinate_systems import CoordinateSystem
from jcm.forcing import ForcingData

from vercor.components import DataComponent, data_component
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid
from vercor.grid_masks import create_lnd_mask_from_ocn
from vercor.setups.data._field_helpers import canonicalize_surface_field

_JCM_LAND_FIELD_NAMES = ("land_surface_temperature", "soil_moisture")


def _jcm_coordinates_in_degrees(
    longitude_radians: ArrayLike,
    latitude_radians: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Convert JCM horizontal coordinates from radians to degrees."""
    return (
        jnp.rad2deg(as_jax_real_array(longitude_radians)),
        jnp.rad2deg(as_jax_real_array(latitude_radians)),
    )


def _prepare_jcm_land_runtime_fields(
    longitude_radians: ArrayLike,
    latitude_radians: ArrayLike,
    land_surface_temperature: ArrayLike,
    soil_moisture: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare JCM land coordinates and monthly forcing fields for VerCOR storage."""
    longitude, latitude = _jcm_coordinates_in_degrees(
        longitude_radians, latitude_radians
    )
    return (
        longitude,
        latitude,
        canonicalize_surface_field(land_surface_temperature),
        canonicalize_surface_field(soil_moisture),
    )


def make_jcm_land(
    jcm_coords: CoordinateSystem,
    jcm_forcing: ForcingData,
    ocn_grid: RectilinearGrid,
    name: str = "LND",
) -> DataComponent:
    """Return a JCM land forcing component."""

    (
        longitude,
        latitude,
        land_surface_temperature,
        soil_moisture,
    ) = _prepare_jcm_land_runtime_fields(
        jcm_coords.horizontal.longitudes,
        jcm_coords.horizontal.latitudes,
        jcm_forcing.stl_am,
        jcm_forcing.soilw_am,
    )
    lnd_bmask, _ = create_lnd_mask_from_ocn(
        atm_lat=latitude,
        atm_lon=longitude,
        ocn_grid=ocn_grid,
    )

    grid = RectilinearGrid(
        name=f"{name.lower()}-grid",
        longitude=longitude,
        latitude=latitude,
        binary_mask=lnd_bmask,
    )

    component = data_component(
        name=name,
        grid=grid,
        fields={
            "land_surface_temperature": land_surface_temperature,
            "soil_moisture": soil_moisture,
        },
    )
    component.declare_fields(outputs=_JCM_LAND_FIELD_NAMES)
    component.update_settings(get_field_time_slice=True)

    return component
