"""Veros host and differentiable JAX runtime stepping helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from vercor._numerical_safety import _require_zero, require_active_finite
from vercor.types import RuntimeArray
from vercor.components import StepContext, StepResult
from vercor.exceptions import ComponentError
import vercor.setups._external.veros_fluxes as _veros_fluxes
import vercor.setups._external.veros_state as _veros_state

if TYPE_CHECKING:
    from vercor.setups._external.veros_gcm_state import VerosGCMSetupState


@jax.jit
def open_water_temperature_contribution(
    sst: RuntimeArray, ice_fraction: RuntimeArray, wet_mask: RuntimeArray
) -> jax.Array:
    """Return (1-A)*current SST in K times ocean-cell area fraction.

    Concentration is validated over every wet ocean cell. Undefined temperature
    over absent water has no effect, including in derivatives. A finite SST
    beneath full ice is retained as an operand so concentration sensitivity is
    not artificially suppressed at that endpoint.
    """
    temperature = jnp.asarray(sst)
    wet = jnp.asarray(wet_mask) > 0
    fraction = jnp.asarray(ice_fraction, dtype=temperature.dtype)
    require_active_finite(fraction, active_mask=wet, owner="Veros ice_fraction")
    fraction = jnp.where(wet, fraction, 0)
    _require_zero(
        jnp.count_nonzero((fraction < 0) | (fraction > 1)),
        "Veros ice_fraction",
        "range [0, 1]",
    )
    require_active_finite(
        temperature, active_mask=wet & (fraction < 1), owner="Veros open-water SST"
    )
    temperature = jnp.where(wet & jnp.isfinite(temperature), temperature, 0)
    return jnp.where(wet, (1 - fraction) * temperature, 0)


def step_veros_runtime(
    resources: "VerosGCMSetupState",
    fields: Mapping[str, Any],
    context: StepContext,
    payload: Any | None,
) -> StepResult:
    """Advance the payload-owned host or JAX Veros ocean boundary."""

    if payload is None:
        raise ComponentError("Veros runtime requires a native runtime payload.")
    execution = resources.execution
    uses_bulk_meteorology = resources.uses_bulk_meteorology
    native_state = payload
    time = context.time
    if execution == "host" and time is None:
        return StepResult(payload=native_state)

    coupled = resources.surface_forcing == "surface_fluxes"
    if coupled:
        taux, tauy, qnet, salt_flux = _veros_fluxes.compute_coupled_fluxes(
            fields, context.constants, context.dtype
        )
        forcing_fields = _veros_state.prepare_surface_forcing_fields(
            taux, tauy, qnet, None, salt_flux
        )
        native_state = _veros_state.apply_veros_forcing_fields(
            native_state,
            forcing_fields,
            execution=execution,
            activate_coupled_feedback=True,
        )
    elif uses_bulk_meteorology:
        taux, tauy, qnet, qnec, salt_flux = _veros_fluxes.compute_fluxes(
            native_state,
            fields,
            context.constants,
            context.dtype,
        )
        forcing_fields = _veros_state.prepare_surface_forcing_fields(
            taux, tauy, qnet, qnec, salt_flux
        )

        native_state = _veros_state.apply_veros_forcing_fields(
            native_state,
            forcing_fields,
            execution=execution,
        )
    native_state = _veros_state.advance_veros_substeps(
        native_state,
        step_function=resources._step_function,
        model_substeps=resources.model_substeps,
        logger=context.logger if execution == "host" else None,
    )
    surface_fields = _veros_state.extract_veros_runtime_surface_fields(
        native_state, coupled=coupled
    )
    if resources.export_open_water_surface:
        surface_fields["open_water_temperature_contribution"] = (
            open_water_temperature_contribution(
                surface_fields["sea_surface_temperature"],
                fields["ice_fraction"],
                resources.grid.binary_mask,
            )
        )
    return StepResult(fields=surface_fields, payload=native_state)


__all__ = ["step_veros_runtime"]
