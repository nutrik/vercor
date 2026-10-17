"""Veros host and differentiable JAX runtime stepping helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from vercor.components import StepContext, StepResult
from vercor.exceptions import ComponentError
import vercor.setups._external.veros_fluxes as _veros_fluxes
import vercor.setups._external.veros_state as _veros_state

if TYPE_CHECKING:
    from vercor.setups._external.veros_gcm_state import VerosGCMSetupState


def step_veros_runtime(
    resources: "VerosGCMSetupState",
    fields: Mapping[str, Any],
    context: StepContext,
    payload: Any | None,
) -> StepResult:
    """Advance the payload-owned host or JAX Veros ocean boundary."""

    if payload is None:
        raise ComponentError("Veros runtime requires a native runtime payload.")
    execution = cast(
        Literal["host", "jax"],
        getattr(resources, "execution", "host"),
    )
    uses_atmosphere_forcing = getattr(resources, "uses_atmosphere_forcing", True)
    native_state = payload
    time = context.time
    if execution == "host" and time is None:
        return StepResult(payload=native_state)

    coupled = getattr(resources, "surface_forcing", "atmosphere") == "coupled"
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
    elif uses_atmosphere_forcing:
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
    return StepResult(
        fields=_veros_state.extract_veros_runtime_surface_fields(
            native_state, coupled=coupled
        ),
        payload=native_state,
    )


__all__ = ["step_veros_runtime"]
