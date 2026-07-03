import jax.numpy as jnp
from typing import TYPE_CHECKING, Any

from vercor.components import DataComponent
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid
from vercor.grid_masks import create_lnd_mask_from_ocn
from vercor.setups.data._field_helpers import canonicalize_surface_field

if TYPE_CHECKING:
    from dinosaur.coordinate_systems import CoordinateSystem
    from jcm.forcing import ForcingData

_JCM_LAND_FIELD_NAMES = ("land_surface_temperature", "soil_moisture")


def make_jcm_land(
    jcm_coords: "CoordinateSystem | Any",
    jcm_forcing: "ForcingData | Any",
    ocn_grid: RectilinearGrid,
    name: str = "LND",
) -> DataComponent:
    """Return a JCM land forcing component."""

    longitude = jnp.rad2deg(as_jax_real_array(jcm_coords.horizontal.longitudes))
    latitude = jnp.rad2deg(as_jax_real_array(jcm_coords.horizontal.latitudes))
    land_surface_temperature = canonicalize_surface_field(jcm_forcing.stl_am)
    soil_moisture = canonicalize_surface_field(jcm_forcing.soilw_am)
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

    component = DataComponent.from_fields(
        name=name,
        grid=grid,
        fields={
            "land_surface_temperature": land_surface_temperature,
            "soil_moisture": soil_moisture,
        },
    )
    component.declare_fields(outputs=_JCM_LAND_FIELD_NAMES)
    component.update_settings(apply_daily_time_selection=True)

    return component
