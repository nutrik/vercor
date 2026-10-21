"""JAXGCM setup-state ownership and lifecycle callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import jax
import jax.numpy as jnp
import tree_math

from dinosaur.coordinate_systems import CoordinateSystem

from jcm.forcing import ForcingData, default_forcing
from jcm.model import Model, ModelPredictions
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData

from vercor.calendar import ModelDateTime
from vercor.components import (
    Component,
    SetupContext,
    SetupResult,
)
from vercor.dtypes import DTypePolicy, as_jax_real_array, jax_ones
from vercor.grids import RectilinearGrid
from vercor.setups._time_helpers import (
    assign_model_timestep_alignment,
    run_logged_spinup,
)
from vercor.setups._external._jax_gcm_pytree import (
    tree_as_runtime_dtype,
    tree_mean,
)
import vercor.setups._external.jax_gcm_runtime as _jax_gcm_runtime
from vercor.setups._external.jax_gcm_tools import change_jcm_parameter_values
from vercor.types import RuntimeArray


@tree_math.struct
@dataclass
class JCMState:
    """JCM output, resumable native state, and its authoritative exact clock."""

    dynamics: PhysicsState
    physics: Any
    dycore_state: Any
    physics_carry: Any
    time: Any
    step: Any


def _bootstrap_jcm_state(model: Model) -> JCMState:
    """Return initialized native dynamics, physics carry, and clock."""

    model.bootstrap_state()
    native = model.run_state
    if native is None or native.dynamics is None or native.physics is None:
        raise RuntimeError("JCM bootstrap did not initialize dycore and physics state")
    return JCMState(
        dynamics=model.dycore.to_physics_state(native.dynamics),
        physics=native.physics,
        dycore_state=native.dynamics,
        physics_carry=native.physics,
        time=native.time,
        step=native.step,
    )


def jax_gcm_epoch(start: datetime | ModelDateTime, effective_spinup: timedelta) -> Any:
    """Bind the native Gregorian clock before any enabled spinup interval."""
    from jcm.date import to_datetime

    if not isinstance(start, datetime):
        raise ValueError("JCM requires a Gregorian datetime start")
    with jax.ensure_compile_time_eval():
        return to_datetime(start - effective_spinup, name="JCM epoch")


class JCMAtmosphereSetupState:
    """Mutable setup-time owner for a JAXGCM/JCM atmosphere adapter."""

    _step_function: Callable[[JCMState, ForcingData], tuple[JCMState, ModelPredictions]]
    _epoch_key: tuple[datetime | ModelDateTime, timedelta, timedelta]
    _state: JCMState
    forcing: ForcingData
    data: dict[str, RuntimeArray]
    coupling_timestep: timedelta
    model_substeps: int

    def __init__(
        self,
        coords: CoordinateSystem,
        terrain: TerrainData,
        name: str = "ATM",
        custom_parameters: Mapping[str, float] | None = None,
        model_timestep: timedelta = timedelta(minutes=30),
        save_interval: timedelta = timedelta(days=1),
        spinup_time: timedelta = timedelta(days=2),
        forcing_data: ForcingData | None = None,
        do_spinup: bool = False,
        jitted: bool = True,
        seaice_coupling: bool = False,
    ) -> None:
        """Build JAXGCM model resources and the VerCOR grid."""

        self.seaice_coupling = seaice_coupling
        self.forcing_data = forcing_data
        self.model_timestep = model_timestep
        self.save_interval = save_interval
        self.spinup_time = spinup_time
        self.do_spinup = do_spinup
        self.jitted = jitted

        jcm_parameters = Parameters.default()

        if custom_parameters is not None:
            change_jcm_parameter_values(
                parameters=custom_parameters,
                default_parameters=jcm_parameters,
            )

        self.open_water_albedo = jcm_parameters.mod_radcon.albsea
        physics = speedy_physics(parameters=jcm_parameters)
        if seaice_coupling:
            from vercor.setups._external.jax_gcm_surface import CoupledSpeedyForcing

            physics = physics.replace(
                "forcing",
                CoupledSpeedyForcing(mod_radcon_params=jcm_parameters.mod_radcon),
            )

        self.model = Model(
            coords=coords,
            time_step=model_timestep.total_seconds() / 60.0,
            terrain=terrain,
            physics=physics,
        )

        hgrid = self.model.coords.horizontal
        grid = RectilinearGrid(
            name=name,
            longitude=jnp.rad2deg(as_jax_real_array(hgrid.longitudes)),
            latitude=jnp.rad2deg(as_jax_real_array(hgrid.latitudes)),
            binary_mask=jax_ones(
                self.model.terrain.fmask.shape
            ).transpose(),  # Interpolation considers every atmosphere point valid.
        )

        self.sigma_levels: RuntimeArray = self.model.coords.vertical.centers

        self.name = name
        self.grid = grid
        self._dtype_policy = DTypePolicy.from_jax_config()

    def _generate_step_function(
        self, jitted: bool = True
    ) -> Callable[[JCMState, ForcingData], tuple[JCMState, ModelPredictions]]:
        """Return the model step function, optionally JIT compiled."""

        def step_function(
            state: JCMState, forcing: ForcingData
        ) -> tuple[JCMState, ModelPredictions]:
            precision_policy = self._dtype_policy
            native, predictions = self.model.run_from_state_with_carry(
                initial_state=state.dycore_state,
                initial_physics_state=state.physics_carry,
                save_interval=self.save_interval / timedelta(days=1),
                total_time=self.coupling_timestep / timedelta(days=1),
                output_averages=False,
                forcing=forcing,
                initial_time=state.time,
                initial_step=state.step,
            )

            # JCM currently returns a stacked object; reduce to one runtime state.
            return (
                JCMState(
                    dynamics=tree_as_runtime_dtype(
                        tree_mean(predictions.dynamics, axis=0),
                        precision_policy,
                    ),
                    physics=tree_as_runtime_dtype(
                        tree_mean(predictions.physics, axis=0),
                        precision_policy,
                    ),
                    dycore_state=native.dynamics,
                    physics_carry=native.physics,
                    time=native.time,
                    step=native.step,
                ),
                predictions,
            )

        return jax.jit(step_function) if jitted else step_function

    def setup(
        self,
        component: Component,
        context: SetupContext,
    ) -> SetupResult:
        """Initialize runtime payload, defaults, and optional spinup state."""

        self._dtype_policy = context.dtype
        assign_model_timestep_alignment(
            self,
            context.dt_seconds,
            self.model_timestep,
        )
        if self.save_interval <= timedelta(0):
            raise ValueError("JCM save_interval must be positive")
        if self.save_interval % self.model_timestep:
            raise ValueError("JCM save_interval must be divisible by model_timestep")
        if self.coupling_timestep % self.save_interval:
            raise ValueError("JCM save_interval must evenly divide coupling_timestep")
        self.spinup_steps = int(
            self.spinup_time.total_seconds() // self.coupling_timestep.total_seconds()
        )

        if context.calendar != "gregorian":
            raise ValueError("JCM 3 requires a Gregorian coupler calendar")
        effective_spinup = self.spinup_time if self.do_spinup else timedelta(0)
        if effective_spinup < timedelta(0) or effective_spinup % self.coupling_timestep:
            raise ValueError("Enabled JCM spinup must align to coupling intervals")
        epoch_key = (context.start, effective_spinup, self.coupling_timestep)
        if hasattr(self, "_epoch_key") and self._epoch_key != epoch_key:
            raise ValueError(
                "JCM epoch already bound; construct a new component for another start"
            )
        self._epoch_key = epoch_key
        self.model.start_time = jax_gcm_epoch(context.start, effective_spinup)

        self._state = _bootstrap_jcm_state(self.model)

        if self.forcing_data is not None:
            self.forcing = self.forcing_data
        else:
            self.forcing = default_forcing(self.model.coords.horizontal)

        self._state = tree_as_runtime_dtype(self._state, self._dtype_policy)
        self.forcing = tree_as_runtime_dtype(self.forcing, self._dtype_policy)
        self._step_function = self._generate_step_function(jitted=self.jitted)

        initial_fields = _jax_gcm_runtime.jax_gcm_default_fields(
            include_total_surface_temperature=False,
        )

        if self.do_spinup:

            def spinup_step(step_number: int) -> None:
                _ = step_number
                _new_state, _predictions = self._step_function(
                    self._state,
                    self.forcing,
                )
                self._state = _new_state
                _ = _predictions

            run_logged_spinup(
                steps=self.spinup_steps,
                logger=context.logger,
                intro_message=f"Performing JCM spinup for {self.spinup_time} day(s)...",
                step_message=lambda step, total: f"JCM spinup step {step} / {total}",
                step=spinup_step,
            )

        payload = _jax_gcm_runtime.create_jax_gcm_runtime_payload(self, component)
        if self.seaice_coupling:
            diagnosed, payload = _jax_gcm_runtime.diagnose_jax_gcm_initial_fields(
                self, payload, context.constants, context.dtype
            )
            initial_fields.update(diagnosed)
        return SetupResult(fields=initial_fields, payload=payload)


__all__ = [
    "JCMAtmosphereSetupState",
    "JCMState",
]
