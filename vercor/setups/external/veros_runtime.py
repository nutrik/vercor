"""Veros host-runtime stepping helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from vercor.components import ComponentStepContext
from vercor.output.variables import OutputVariable
import vercor.setups.external.veros_fluxes as _veros_fluxes
import vercor.setups.external.veros_output as _veros_output
import vercor.setups.external.veros_state as _veros_state

if TYPE_CHECKING:
    from vercor.setups.external.veros_gcm_state import VerosGCMSetupState


def step_veros_runtime(
    state: "VerosGCMSetupState",
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
    state: "VerosGCMSetupState",
    context: ComponentStepContext,
) -> None:
    """Record selected Veros variables and write optional period output."""

    output_variables = getattr(state, "output_variables", ())
    if not output_variables:
        return

    time = context.time
    if time is None:
        return

    state.output_adapter.accumulate(
        _veros_output.extract_veros_output_snapshot(
            state._veros_state,
            output_variables,
        )
    )

    def build_coordinate_variables(
        variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        return _veros_output.veros_average_coordinate_variables(
            veros_state=state._veros_state,
            output_time=time,
            variables=variables,
        )

    state.output_adapter.write_period_average_if_due(
        time=time,
        dt=timedelta(seconds=context.dt_seconds),
        output_frequency=state.output_frequency,
        output=lambda output_time: f"veros.averages.{output_time.strftime('%Y-%m-%d')}.nc",
        build_coordinate_variables=build_coordinate_variables,
        logger=context.logger,
    )


__all__ = ["record_veros_output", "step_veros_runtime"]
