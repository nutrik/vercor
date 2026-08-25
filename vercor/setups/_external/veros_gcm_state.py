"""Veros setup-state ownership and lifecycle callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from functools import partial
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp

from vercor.components import (
    Component,
    SetupContext,
    SetupResult,
)
from vercor.grids import RectilinearGrid
from vercor.setups._time_helpers import (
    assign_model_timestep_alignment,
    run_logged_spinup,
)
import vercor.setups._external.veros_setup as _veros_setup
import vercor.setups._external.veros_state as _veros_state
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


def _select_veros_setup_class(
    setup: Literal["global_4deg", "acc"],
    execution: Literal["host", "jax"],
) -> type[Any]:
    """Return the native setup class for one execution lane."""

    if execution == "jax":
        from vercor.setups._external.veros_setup_jax import (
            DifferentiableACCSetup,
            DifferentiableGlobalFourDegree,
        )

        return (
            DifferentiableGlobalFourDegree
            if setup == "global_4deg"
            else DifferentiableACCSetup
        )
    if setup == "global_4deg":
        return _veros_setup.CustomGlobalFourDegree
    from veros.setups.acc import ACCSetup

    return cast(type[Any], ACCSetup)


class VerosGCMSetupState:
    """Mutable setup-time owner for one host or JAX Veros ocean adapter."""

    name: str
    data: dict[str, RuntimeArray]
    coupling_timestep: timedelta
    model_timestep: timedelta
    model_substeps: int
    _linear_solver: Any
    _step_function: Callable[[Any], Any]

    def __init__(
        self,
        name: str = "OCN",
        setup: Literal["global_4deg", "acc"] = "global_4deg",
        execution: Literal["host", "jax"] = "host",
        spinup_time: timedelta = timedelta(days=2),
        custom_parameters: Mapping[str, Any] | None = None,
        restore_to_climatology: bool = False,
        do_spinup: bool = False,
        jitted: bool = False,
    ) -> None:
        """Build Veros model resources and the VerCOR ocean grid."""

        self.name = name
        self.execution = execution
        self.uses_atmosphere_forcing = setup == "global_4deg"
        override = dict(custom_parameters or {})
        if execution == "jax":
            override["enable_streamfunction"] = False

        setup_class = _select_veros_setup_class(setup, execution)
        self.model = setup_class(override=override)
        self.model.setup()
        self._linear_solver = _veros_state.get_component_linear_solver(self.model.state)
        native_jitted = jitted or execution == "jax"
        self._veros_state = _veros_state.copy_state(
            self.model.state,
            jitted=native_jitted,
            execution=execution,
        )
        step_function = cast(
            Callable[[Any], Any],
            partial(
                _veros_state.pure,
                jitted=native_jitted,
                step=self.model.step,
                linear_solver=self._linear_solver,
                execution=execution,
            ),
        )
        self._step_function = (
            cast(Callable[[Any], Any], jax.jit(step_function))
            if execution == "jax"
            else step_function
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

    def setup(
        self,
        component: Component,
        context: SetupContext,
    ) -> SetupResult:
        """Align timestep, optionally spin up, and seed the initial SST."""

        dt_seconds = context.dt_seconds
        assign_model_timestep_alignment(
            self,
            dt_seconds,
            timedelta(seconds=float(self.dt_tracer)),
            coupling_name="dt",
            model_name="dt_tracer",
        )

        if self.do_spinup:

            def spinup_step(step_number: int) -> None:
                _ = step_number
                self._veros_state = self._step_function(self._veros_state)

            run_logged_spinup(
                steps=self.spinup_steps,
                logger=context.logger,
                intro_message=f"Performing Veros spinup for {self.spinup_time} day(s)...",
                step_message=lambda step, total: f"Step {step} / {total}",
                step=spinup_step,
            )

        _ = component
        return SetupResult(
            fields={
                "sea_surface_temperature": _veros_state.extract_veros_runtime_sst(
                    self._veros_state
                )
            },
            payload=self._veros_state,
        )


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
    "veros_default_fields",
]
