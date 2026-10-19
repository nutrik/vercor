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

_LAND_INPUTS = ("latent_heat_flux", "sensible_heat_flux")
_LAND_OUTPUTS = ("soil_moisture", "land_surface_temperature")
_LAND_DEFAULT_FIELDS = {
    "soil_moisture": 0.3,
    "land_surface_temperature": 288.15,
    "latent_heat_flux": 0.0,
    "sensible_heat_flux": 0.0,
}


@jax.jit
def _update_soil_moisture(
    soil_moisture: object,
    latent_heat_flux: object,
    dt_seconds: float,
) -> jax.Array:
    """Advance dimensionless moisture using downward-positive latent heat.

    Negative W/m2 flux dries soil; positive flux wets it over dt_seconds.
    The retained 1e-9 coefficient is a toy closure, not latent-heat physics.
    Grid-shaped moisture is clipped to [0, 1].
    """
    soil_moisture_array = as_jax_real_array(soil_moisture)
    latent_heat_flux_array = as_jax_real_array(latent_heat_flux)
    moisture_tendency = 1e-9 * latent_heat_flux_array
    return jnp.clip(soil_moisture_array + moisture_tendency * dt_seconds, 0.0, 1.0)


def make_slab_land(
    grid: RectilinearGrid,
    name: str = "LND",
    *,
    output: OutputSpec | None = None,
    fractional_coverage: object | None = None,
) -> Component:
    """Return a toy land component with optional grid-shaped surface coverage.

    Coverage is dimensionless, in [0, 1], within the grid's static support.
    The runtime validates and applies it to exchanges; local temperatures and
    flux-driven tendencies remain per unit surface area.
    """

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

    initial_fields: dict[str, object] = dict(_LAND_DEFAULT_FIELDS)
    masks = None
    if fractional_coverage is not None:
        initial_fields["surface_fraction"] = fractional_coverage
        masks = ComponentMaskSpec(fractional_field="surface_fraction")

    return CallableComponent(
        name,
        grid,
        step,
        spec=ComponentSpec(
            inputs=_LAND_INPUTS,
            outputs=_LAND_OUTPUTS + (("surface_fraction",) if masks else ()),
            initial_fields=initial_fields,
            masks=masks,
            output=resolve_output(output),
        ),
    )
