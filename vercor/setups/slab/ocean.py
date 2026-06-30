from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from vercor.components import (
    Component,
    ComponentStepContext,
)
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid

_REFERENCE_SEA_SURFACE_TEMPERATURE = 273.15 + 15.0
_OCEAN_INPUTS = ("sensible_heat_flux", "latent_heat_flux")
_OCEAN_OUTPUTS = ("sea_surface_temperature",)
_OCEAN_DEFAULT_FIELDS = {"sea_surface_temperature": _REFERENCE_SEA_SURFACE_TEMPERATURE}


@jax.jit
def _advance_sea_surface_temperature(
    sea_surface_temperature: object,
    sensible_heat_flux: object,
    latent_heat_flux: object,
    dt_seconds: float,
    rho: float,
    cp: float,
    mixed_layer_depth: float,
    lambda_relax: float,
    reference_temperature: float,
) -> jax.Array:
    sea_surface_temperature_array = as_jax_real_array(sea_surface_temperature)
    sensible_heat_flux_array = as_jax_real_array(sensible_heat_flux)
    latent_heat_flux_array = as_jax_real_array(latent_heat_flux)
    qnet = sensible_heat_flux_array + latent_heat_flux_array
    tendency = qnet / (rho * cp * mixed_layer_depth) + lambda_relax * (
        sea_surface_temperature_array - reference_temperature
    )
    return sea_surface_temperature_array + tendency * dt_seconds


def make_slab_ocean(
    grid: RectilinearGrid,
    name: str = "OCN",
    H: float = 30.0,
) -> Component:
    """Return a toy slab ocean component factory instance."""

    rho = 1025.0
    cp = 3990.0
    lambda_relax = 1.0 / (30.0 * 86400.0)

    def step(
        fields: Mapping[str, Any],
        context: ComponentStepContext,
    ) -> Mapping[str, Any]:
        dt_seconds = context.dt_seconds
        sea_surface_temperature = fields["sea_surface_temperature"]
        sensible_heat_flux = fields.get(
            "sensible_heat_flux",
            jnp.zeros_like(as_jax_real_array(sea_surface_temperature)),
        )
        latent_heat_flux = fields.get(
            "latent_heat_flux",
            jnp.zeros_like(as_jax_real_array(sea_surface_temperature)),
        )

        updated_sst = _advance_sea_surface_temperature(
            sea_surface_temperature,
            sensible_heat_flux,
            latent_heat_flux,
            dt_seconds,
            rho,
            cp,
            H,
            lambda_relax,
            _REFERENCE_SEA_SURFACE_TEMPERATURE,
        )
        return {"sea_surface_temperature": updated_sst}

    return Component.from_model(
        name=name,
        grid=grid,
        step=step,
        inputs=_OCEAN_INPUTS,
        outputs=_OCEAN_OUTPUTS,
        default_fields=_OCEAN_DEFAULT_FIELDS,
    )
