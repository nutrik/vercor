from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from vercor.components.base import Component, differentiable_component
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid

_SEAICE_INPUTS = ("sea_surface_temperature",)
_SEAICE_OUTPUTS = ("ice_fraction",)
_SEAICE_DEFAULT_FIELDS = {"ice_fraction": 0.0}


@jax.jit
def _diagnose_ice_fraction(sea_surface_temperature: object) -> jax.Array:
    sea_surface_temperature_array = as_jax_real_array(sea_surface_temperature)
    freezing_temperature = 273.15 - 1.8
    x = (freezing_temperature - sea_surface_temperature_array) / 2.0
    return 1.0 / (1.0 + jnp.exp(-x))


def make_slab_seaice(grid: RectilinearGrid, name: str = "ICE") -> Component:
    """Return a toy thermodynamic sea-ice component factory instance."""

    def step(fields: Mapping[str, Any]) -> Mapping[str, Any]:
        sea_surface_temperature = fields.get("sea_surface_temperature")
        if sea_surface_temperature is None:
            return {}
        ice_fraction = _diagnose_ice_fraction(sea_surface_temperature)
        return {"ice_fraction": ice_fraction}

    return differentiable_component(
        name=name,
        grid=grid,
        step=step,
        inputs=_SEAICE_INPUTS,
        outputs=_SEAICE_OUTPUTS,
        default_fields=_SEAICE_DEFAULT_FIELDS,
    )
