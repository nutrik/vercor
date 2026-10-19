from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from vercor.components import (
    ComponentMaskSpec,
    ComponentSpec,
    DataComponent,
    TransferPolicy,
)
from vercor.dtypes import as_jax_real_array
from vercor.grids import RectilinearGrid
from vercor.grid_masks import create_lnd_mask_from_ocn
from vercor.output import OutputSpec
from vercor.setups._output import resolve_output

if TYPE_CHECKING:
    from dinosaur.coordinate_systems import CoordinateSystem as _CoordinateSystem
    from jcm.forcing import ForcingData as _ForcingData
else:
    _CoordinateSystem = Any
    _ForcingData = Any

_JCM_LAND_INPUT_NAMES: tuple[str, ...] = ()
_JCM_LAND_FIELD_NAMES = ("land_surface_temperature", "soil_moisture", "land_fraction")


def _canonicalize_jcm_forcing_field(
    field: Any,
    *,
    field_name: str,
    expected_spatial_shape: tuple[int, int],
) -> jax.Array:
    """Return one static or time-first JCM surface field in VerCOR layout."""

    values = getattr(field, "values", field)
    array = as_jax_real_array(values)
    if array.ndim not in (2, 3):
        raise ValueError(
            f"JCM forcing field '{field_name}' has shape {array.shape}; expected "
            "(longitude, latitude) or (time, longitude, latitude)"
        )
    if array.shape[-2:] != expected_spatial_shape:
        raise ValueError(
            f"JCM forcing field '{field_name}' has shape {array.shape}; expected "
            "spatial dimensions (longitude, latitude) "
            f"{expected_spatial_shape}"
        )
    return array.T if array.ndim == 2 else array.transpose((0, 2, 1))


def make_jcm_land(
    jcm_coords: _CoordinateSystem | Any,
    jcm_forcing: _ForcingData | Any,
    ocn_grid: RectilinearGrid,
    name: str = "LND",
    *,
    output: OutputSpec | None = None,
) -> DataComponent:
    """Return a JCM land forcing component."""

    longitude = jnp.rad2deg(as_jax_real_array(jcm_coords.horizontal.longitudes))
    latitude = jnp.rad2deg(as_jax_real_array(jcm_coords.horizontal.latitudes))
    forcing_spatial_shape = (longitude.shape[0], latitude.shape[0])
    land_surface_temperature = _canonicalize_jcm_forcing_field(
        jcm_forcing.stl_am,
        field_name="stl_am",
        expected_spatial_shape=forcing_spatial_shape,
    )
    soil_moisture = _canonicalize_jcm_forcing_field(
        jcm_forcing.soilw_am,
        field_name="soilw_am",
        expected_spatial_shape=forcing_spatial_shape,
    )
    daily = land_surface_temperature.ndim == 3 or soil_moisture.ndim == 3
    if daily:
        records = max(
            (
                land_surface_temperature.shape[0]
                if land_surface_temperature.ndim == 3
                else 1
            ),
            soil_moisture.shape[0] if soil_moisture.ndim == 3 else 1,
        )
        shape = (records, *land_surface_temperature.shape[-2:])
        land_surface_temperature = jnp.broadcast_to(land_surface_temperature, shape)
        soil_moisture = jnp.broadcast_to(soil_moisture, shape)
    lnd_bmask, land_fraction = create_lnd_mask_from_ocn(
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

    component = DataComponent(
        name,
        grid,
        {
            "land_surface_temperature": land_surface_temperature,
            "soil_moisture": soil_moisture,
            "land_fraction": land_fraction,
        },
        spec=ComponentSpec(
            inputs=_JCM_LAND_INPUT_NAMES,
            outputs=_JCM_LAND_FIELD_NAMES,
            initial_fields={field_name: 0.0 for field_name in _JCM_LAND_INPUT_NAMES},
            transfer=TransferPolicy(time_selection="daily" if daily else "current"),
            masks=ComponentMaskSpec(fractional_field="land_fraction"),
            output=resolve_output(output),
        ),
    )

    return component
