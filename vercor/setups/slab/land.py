from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from vercor.components import (
    Component,
    StepContext,
)
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid

_LAND_INPUTS = ("latent_heat_flux",)
_LAND_OUTPUTS = ("soil_moisture", "land_surface_temperature")
_LAND_DEFAULT_FIELDS = {"soil_moisture": 0.3, "land_surface_temperature": 288.15}


@jax.jit
def _update_soil_moisture(
    soil_moisture: object,
    latent_heat_flux: object,
    dt_seconds: float,
) -> jax.Array:
    soil_moisture_array = as_jax_real_array(soil_moisture)
    latent_heat_flux_array = as_jax_real_array(latent_heat_flux)
    evap = 1e-9 * latent_heat_flux_array
    return jnp.clip(soil_moisture_array - evap * dt_seconds, 0.0, 1.0)


def make_slab_land(grid: RectilinearGrid, name: str = "LND") -> Component:
    """Return a toy bucket land component factory instance."""

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        dt_seconds = context.dt_seconds
        soil_moisture = fields["soil_moisture"]
        latent_heat_flux = fields.get(
            "latent_heat_flux",
            jnp.zeros_like(as_jax_real_array(soil_moisture)),
        )
        updated_soil_moisture = _update_soil_moisture(
            soil_moisture,
            latent_heat_flux,
            dt_seconds,
        )
        return {"soil_moisture": updated_soil_moisture}

    return Component.from_step(
        name=name,
        grid=grid,
        step=step,
        inputs=_LAND_INPUTS,
        outputs=_LAND_OUTPUTS,
        defaults=_LAND_DEFAULT_FIELDS,
    )
