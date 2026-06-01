"""Veros setup-state ownership and lifecycle callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from functools import partial
from typing import Any, cast

import jax.numpy as jnp

from vercor.components import (
    ComponentSetupContext,
    ComponentStepContext,
    HostRuntimeComponent,
)
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.setups._time_helpers import (
    assign_model_timestep_alignment,
    run_logged_spinup,
)
import vercor.setups.external.veros_runtime as _veros_runtime
import vercor.setups.external.veros_setup as _veros_setup
import vercor.setups.external.veros_state as _veros_state
from vercor.types import RuntimeArray

VEROS_INPUT_FIELD_NAMES = (
    "model_level_height",
    "u_velocity",
    "v_velocity",
    "potential_temperature",
    "specific_humidity",
    "density",
    "temperature",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
VEROS_FIELD_DEFAULTS = {"sea_surface_temperature": 283.15}


def advance_veros_model_step(
    veros_state: Any,
    *,
    step: Callable[[Any], Any],
    jitted: bool,
) -> Any:
    """Advance a Veros state through the configured host step boundary."""

    return _veros_state.pure(veros_state, jitted=jitted, step=step)


class VerosGCMSetupState:
    """Mutable setup-time owner for a host-backed Veros ocean adapter."""

    name: str
    data: dict[str, RuntimeArray]
    settings: VercorSettings
    coupling_timestep: timedelta
    model_timestep: timedelta
    model_substeps: int
    _step_function: Callable[[Any], Any]

    def __init__(
        self,
        name: str = "OCN",
        spinup_time: timedelta = timedelta(days=2),
        custom_parameters: dict[str, Any] | None = None,
        restore_to_climatology: bool = False,
        do_spinup: bool = False,
        jitted: bool = False,
    ) -> None:
        """Build Veros model resources and the VerCOR ocean grid."""

        self.name = name
        override = custom_parameters or {}

        self.model = _veros_setup.CustomGlobalFourDegree(override=override)
        self.model.setup()
        self._veros_state = _veros_state.copy_state(self.model.state, jitted=jitted)
        self._step_function = cast(
            Callable[[Any], Any],
            partial(
                advance_veros_model_step,
                step=self.model.step,
                jitted=jitted,
            ),
        )

        self.do_spinup = do_spinup
        self.spinup_time = spinup_time
        self.restore_to_climatology = restore_to_climatology
        self.jitted = jitted

        self.dt_tracer = getattr(self._veros_state.settings, "dt_tracer")
        self.spinup_steps = int(self.spinup_time.total_seconds() // self.dt_tracer)

        mask = jnp.where(
            jnp.asarray(self._veros_state.variables.maskT[:, :, -1]) > 0.0,
            1.0,
            0.0,
        )

        self.grid = RectilinearGrid(
            name=name,
            longitude=self._veros_state.variables.xt[2:-2],
            latitude=self._veros_state.variables.yt[2:-2],
            binary_mask=mask[2:-2, 2:-2].T,
        )

    def initialize(
        self,
        component: HostRuntimeComponent,
        context: ComponentSetupContext,
    ) -> None:
        """Align timestep, optionally spin up, and seed the initial SST."""

        dt_seconds = context.dt_seconds
        assign_model_timestep_alignment(
            self,
            dt_seconds,
            timedelta(seconds=float(self.dt_tracer)),
            coupling_name="dt",
            model_name="dt_tracer",
        )

        if self.do_spinup and "ATM" in context.run_sequence.order:

            def spinup_step(step_number: int) -> None:
                _ = step_number
                self._veros_state = self._step_function(self._veros_state)

            run_logged_spinup(
                steps=self.spinup_steps,
                logger=context.logger,
                intro_message=f" Performing Veros spinup for {self.spinup_time} day(s)...",
                step_message=lambda step, total: f" Step {step} / {total}",
                step=spinup_step,
            )

        component.seed_field(
            "sea_surface_temperature",
            _veros_state.extract_veros_runtime_sst(self._veros_state),
        )

    def step(
        self,
        fields: Mapping[str, Any],
        context: ComponentStepContext,
        payload: Any | None,
    ) -> Mapping[str, Any]:
        """Delegate Veros host-runtime advancement to the runtime helper."""

        return _veros_runtime.step_veros_runtime(self, fields, context, payload)


def veros_default_fields() -> dict[str, float]:
    """Return scalar defaults for the Veros runtime output contract."""

    return {
        field_name: VEROS_FIELD_DEFAULTS.get(field_name, 0.0)
        for field_name in ("sea_surface_temperature",)
    }


__all__ = [
    "VEROS_FIELD_DEFAULTS",
    "VEROS_INPUT_FIELD_NAMES",
    "VerosGCMSetupState",
    "advance_veros_model_step",
    "veros_default_fields",
]
