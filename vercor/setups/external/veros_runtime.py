"""Veros host-runtime stepping helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Any, Protocol

from vercor.components import ComponentStepContext
from vercor.setups.external.jax_gcm_output import should_write_period_output
from vercor.setups.external.veros_output import (
    accumulate_veros_output_snapshot,
    extract_veros_output_snapshot,
    write_veros_averages_output,
)
from vercor.setups.external.period_averages import PeriodAverageAccumulator
import vercor.setups.external.veros_fluxes as _veros_fluxes
import vercor.setups.external.veros_state as _veros_state


class _VerosRuntimeState(Protocol):
    """Private protocol for the Veros setup state consumed by runtime hooks."""

    _veros_state: Any
    _step_function: Callable[[Any], Any]
    restore_to_climatology: bool
    jitted: bool
    model_substeps: int
    output_frequency: str | None
    output_variables: tuple[str, ...]
    _period_average_accumulator: PeriodAverageAccumulator


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
    forcing_fields = _veros_state.prepare_surface_forcing_fields(
        taux, tauy, qnet, qnec, state.restore_to_climatology
    )

    state._veros_state = _veros_state.apply_veros_forcing_fields(
        state._veros_state,
        forcing_fields,
        jitted=state.jitted,
    )
    state._veros_state = _veros_state.advance_veros_substeps(
        state._veros_state,
        step_function=state._step_function,
        model_substeps=state.model_substeps,
        logger=logger,
    )
    record_veros_output(state, context)

    return {
        "sea_surface_temperature": _veros_state.extract_veros_runtime_sst(
            state._veros_state
        )
    }


def record_veros_output(
    state: _VerosRuntimeState,
    context: ComponentStepContext,
) -> None:
    """Record selected Veros variables and write optional period output."""

    output_variables = getattr(state, "output_variables", ())
    if not output_variables:
        return

    time = context.time
    if time is None:
        return

    accumulate_veros_output_snapshot(
        state._period_average_accumulator,
        extract_veros_output_snapshot(state._veros_state, output_variables),
    )
    if should_write_period_output(
        time=time,
        dt=timedelta(seconds=context.dt_seconds),
        output_frequency=state.output_frequency,
    ):
        date_time = time.strftime("%Y-%m-%d")
        write_veros_averages_output(
            state._period_average_accumulator,
            output=f"veros.averages.{date_time}.nc",
            veros_state=state._veros_state,
            output_time=time,
            logger=context.logger,
        )


__all__ = ["record_veros_output", "step_veros_runtime"]
