"""JAXGCM setup-state ownership and lifecycle callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
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
    grid_field_defaults,
)
from vercor.setups._external._jax_gcm_pytree import (
    tree_as_runtime_dtype,
    tree_mean,
)
import vercor.setups._external.jax_gcm_fields as _jax_gcm_fields
import vercor.setups._external.jax_gcm_runtime as _jax_gcm_runtime
from vercor.setups._external.jax_gcm_tools import change_jcm_parameter_values
from vercor.types import RuntimeArray


@tree_math.struct
@dataclass
class JCMState:
    """JCM gridpoint output, native dycore state, and physics carry."""

    dynamics: PhysicsState
    physics: Any
    dycore_state: Any
    physics_carry: Any


def _bootstrap_jcm_state(model: Model) -> JCMState:
    """Return the initialized JCM 2 dycore and physics state bundle."""

    model.bootstrap_state()
    dycore_state = model._final_dycore_state
    physics_carry = model._final_physics_state
    if dycore_state is None or physics_carry is None:
        raise RuntimeError("JCM bootstrap did not initialize dycore and physics state")
    return JCMState(
        dynamics=model.dycore.to_physics_state(dycore_state),
        physics=physics_carry,
        dycore_state=dycore_state,
        physics_carry=physics_carry,
    )


class JAXGCMSetupState:
    """Mutable setup-time owner for a JAXGCM/JCM atmosphere adapter."""

    _step_function: Callable[[JCMState, ForcingData], tuple[JCMState, ModelPredictions]]
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
    ) -> None:
        """Build JAXGCM model resources and the VerCOR grid."""

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

        physics = speedy_physics(parameters=jcm_parameters)

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
            final_dycore_state, final_physics_carry, predictions = (
                self.model.run_from_state_with_carry(
                    initial_state=state.dycore_state,
                    initial_physics_state=state.physics_carry,
                    save_interval=self.save_interval / timedelta(days=1),
                    total_time=self.coupling_timestep / timedelta(days=1),
                    output_averages=False,
                    forcing=forcing,
                )
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
                    dycore_state=final_dycore_state,
                    physics_carry=final_physics_carry,
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
        self.spinup_steps = int(
            self.spinup_time.total_seconds() // self.coupling_timestep.total_seconds()
        )

        self._state = _bootstrap_jcm_state(self.model)

        if self.forcing_data is not None:
            self.forcing = self.forcing_data
        else:
            self.forcing = default_forcing(self.model.coords.horizontal)

        self._state = tree_as_runtime_dtype(self._state, self._dtype_policy)
        self.forcing = tree_as_runtime_dtype(self.forcing, self._dtype_policy)
        self._step_function = self._generate_step_function(jitted=self.jitted)

        initial_fields = grid_field_defaults(
            _jax_gcm_runtime.jax_gcm_default_field_names(
                include_total_surface_temperature=False,
            ),
            overrides={
                "sea_surface_temperature": (
                    _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE
                ),
            },
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

        return SetupResult(
            fields=initial_fields,
            payload=_jax_gcm_runtime.create_jax_gcm_runtime_payload(self, component),
        )


__all__ = [
    "JAXGCMSetupState",
    "JCMState",
]
