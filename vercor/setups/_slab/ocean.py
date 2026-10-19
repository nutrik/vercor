from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from vercor.components import (
    CallableComponent,
    Component,
    ComponentSpec,
    ComponentMaskSpec,
    StepContext,
)
from vercor.dtypes import as_jax_real_array
from vercor.grids import RectilinearGrid
from vercor.output import OutputSpec
from vercor.setups._output import resolve_output

_REFERENCE_SEA_SURFACE_TEMPERATURE = 273.15 + 15.0
_OCEAN_INPUTS = (
    "sensible_heat_flux",
    "latent_heat_flux",
    "u_velocity_10m",
    "v_velocity_10m",
    "u_velocity",
    "v_velocity",
    "specific_humidity",
    "temperature",
    "model_level_height",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "ice_fraction",
)
_OCEAN_OUTPUTS = ("sea_surface_temperature",)
_OCEAN_DEFAULT_FIELDS = {
    "sea_surface_temperature": _REFERENCE_SEA_SURFACE_TEMPERATURE,
    "sensible_heat_flux": 0.0,
    "latent_heat_flux": 0.0,
    "u_velocity_10m": 0.0,
    "v_velocity_10m": 0.0,
    "u_velocity": 0.0,
    "v_velocity": 0.0,
    "specific_humidity": 0.0,
    "temperature": 0.0,
    "model_level_height": 0.0,
    "net_shortwave_radiation_flux": 0.0,
    "downward_longwave_radiation_flux": 0.0,
    "ice_fraction": 0.0,
}


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
    """Advance grid-shaped SST in K with downward-positive heat and restoring.

    Heat fluxes are in W/m2, rho in kg/m3, cp in J/(kg K), depth in m and
    lambda_relax in 1/s. Explicit Euler damps SST toward reference_temperature
    for positive lambda_relax and dt_seconds below the restoring timescale.
    """
    sea_surface_temperature_array = as_jax_real_array(sea_surface_temperature)
    sensible_heat_flux_array = as_jax_real_array(sensible_heat_flux)
    latent_heat_flux_array = as_jax_real_array(latent_heat_flux)
    qnet = sensible_heat_flux_array + latent_heat_flux_array
    tendency = qnet / (rho * cp * mixed_layer_depth) - lambda_relax * (
        sea_surface_temperature_array - reference_temperature
    )
    return sea_surface_temperature_array + tendency * dt_seconds


def make_slab_ocean(
    grid: RectilinearGrid,
    name: str = "OCN",
    mixed_layer_depth: float = 30.0,
    *,
    output: OutputSpec | None = None,
    fractional_coverage: object | None = None,
) -> Component:
    """Return a toy ocean component with optional grid-shaped surface coverage.

    Coverage is dimensionless, in [0, 1], within the grid's static support.
    The runtime validates and applies it to exchanges; local temperatures and
    flux-driven tendencies remain per unit surface area.
    """

    rho = 1025.0
    cp = 3990.0
    lambda_relax = 1.0 / (30.0 * 86400.0)

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
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
            mixed_layer_depth,
            lambda_relax,
            _REFERENCE_SEA_SURFACE_TEMPERATURE,
        )
        return {"sea_surface_temperature": updated_sst}

    initial_fields: dict[str, object] = dict(_OCEAN_DEFAULT_FIELDS)
    masks = None
    if fractional_coverage is not None:
        initial_fields["surface_fraction"] = fractional_coverage
        masks = ComponentMaskSpec(fractional_field="surface_fraction")

    return CallableComponent(
        name,
        grid,
        step,
        spec=ComponentSpec(
            inputs=_OCEAN_INPUTS,
            outputs=_OCEAN_OUTPUTS + (("surface_fraction",) if masks else ()),
            initial_fields=initial_fields,
            masks=masks,
            output=resolve_output(output),
        ),
    )
