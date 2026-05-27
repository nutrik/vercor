"""Veros host-runtime stepping helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from vercor.components import ComponentStepContext
import vercor.setups.external.veros_fluxes as _veros_fluxes
import vercor.setups.external.veros_state as _veros_state


class _VerosRuntimeState(Protocol):
    """Private protocol for the Veros setup state consumed by runtime hooks."""

    _veros_state: Any
    _step_function: Callable[[Any], Any]
    restore_to_climatology: bool
    jitted: bool
    model_substeps: int


def step_veros_runtime(
    state: _VerosRuntimeState,
    fields: Mapping[str, Any],
    context: ComponentStepContext,
    payload: Any | None,
) -> Mapping[str, Any]:
    """Advance the private host-backed Veros ocean boundary."""

    _ = payload
    time = context.time
    logger = context.logger
    if time is None:
        return {}

    taux, tauy, qnet, qnec = _veros_fluxes.compute_fluxes(
        state._veros_state,
        fields,
        context.settings,
    )
    forcing_fields = _veros_state._prepare_surface_forcing_fields(
        taux, tauy, qnet, qnec, state.restore_to_climatology
    )

    state._veros_state = _veros_state._apply_veros_forcing_fields(
        state._veros_state,
        forcing_fields,
        jitted=state.jitted,
    )
    state._veros_state = _veros_state._advance_veros_substeps(
        state._veros_state,
        step_function=state._step_function,
        model_substeps=state.model_substeps,
        logger=logger,
    )

    return {
        "sea_surface_temperature": _veros_state._extract_veros_runtime_sst(
            state._veros_state
        )
    }


__all__ = ["step_veros_runtime"]
