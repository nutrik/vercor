"""JAXGCM setup-state ownership and lifecycle callbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import jax
import jax.numpy as jnp
import tree_math

from dinosaur import primitive_equations
from dinosaur.coordinate_systems import CoordinateSystem

from jcm.forcing import default_forcing
from jcm.model import ForcingData, Model, Predictions
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.physics_interface import (
    PhysicsState,
    TerrainData,
    dynamics_state_to_physics_state,
)

from vercor.components import (
    Component,
    ComponentSetupContext,
)
from vercor.dtypes import as_jax_real_array, jax_ones
from vercor.grid import RectilinearGrid
from vercor.output.adapters import ComponentOutputAdapter
from vercor.pytree_utils import asfloat, mean_leaf
from vercor.setups._time_helpers import (
    assign_model_timestep_alignment,
    run_logged_spinup,
    seed_grid_field_defaults,
)
import vercor.setups.external.jax_gcm_fields as _jax_gcm_fields
import vercor.setups.external.jax_gcm_output as _jax_gcm_output
import vercor.setups.external.jax_gcm_runtime as _jax_gcm_runtime
from vercor.setups.external.jax_gcm_tools import change_jcm_parameter_values
from vercor.types import RuntimeArray


@tree_math.struct
@dataclass
class JCMState:
    """JCM prognostic, physics, and primitive-equation state bundle."""

    prog: PhysicsState
    phydata: Any
    metadata: primitive_equations.State


class JAXGCMSetupState:
    """Mutable setup-time owner for a JAXGCM/JCM atmosphere adapter."""

    output_adapter: ComponentOutputAdapter
    _step_function: Callable[[JCMState, ForcingData], tuple[JCMState, Predictions]]
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
        custom_parameters: dict[str, float] | None = None,
        model_timestep: timedelta = timedelta(minutes=30),
        save_interval: timedelta = timedelta(days=1),
        spinup_time: timedelta = timedelta(days=2),
        forcing_data: ForcingData | None = None,
        output_frequency: str | None = None,
        do_spinup: bool = False,
        jitted: bool = True,
    ) -> None:
        """Build JAXGCM model resources and the VerCOR grid."""

        self.forcing_data = forcing_data
        self.output_frequency = output_frequency
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

        physics = SpeedyPhysics(parameters=jcm_parameters)

        self.model = Model(
            coords,
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
        self.settings: Any | None = None
        self.output_adapter = _jax_gcm_output.make_jax_gcm_output_adapter()

    def _generate_step_function(
        self, jitted: bool = True
    ) -> Callable[[JCMState, ForcingData], tuple[JCMState, Predictions]]:
        """Return the model step function, optionally JIT compiled."""

        def step_function(
            state: JCMState, forcing: ForcingData
        ) -> tuple[JCMState, Predictions]:
            precision_policy = getattr(self, "settings", None)
            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state=state.metadata,
                save_interval=self.save_interval / timedelta(days=1),
                total_time=self.coupling_timestep / timedelta(days=1),
                forcing=forcing,
            )

            # JCM currently returns a stacked object; reduce to one runtime state.
            return (
                JCMState(
                    prog=asfloat(
                        mean_leaf(predictions.dynamics, axis=0),
                        precision_policy,
                    ),
                    phydata=asfloat(
                        mean_leaf(predictions.physics, axis=0),
                        precision_policy,
                    ),
                    metadata=new_atm_modal_state,
                ),
                predictions,
            )

        return jax.jit(step_function) if jitted else step_function

    def initialize(
        self,
        component: Component,
        context: ComponentSetupContext,
    ) -> None:
        """Initialize runtime payload, defaults, and optional spinup state."""

        self.settings = context.settings
        assign_model_timestep_alignment(
            self,
            context.dt_seconds,
            self.model_timestep,
        )
        self.spinup_steps = int(
            self.spinup_time.total_seconds() // self.coupling_timestep.total_seconds()
        )

        _modal_state = self.model._prepare_initial_modal_state()
        self._state = JCMState(
            metadata=_modal_state,
            phydata=PhysicsData.zeros(
                self.model.coords.horizontal.nodal_shape,
                self.model.coords.vertical.layers,
            ),
            prog=dynamics_state_to_physics_state(_modal_state, self.model.primitive),
        )

        if self.forcing_data is not None:
            self.forcing = self.forcing_data
        else:
            self.forcing = default_forcing(self.model.coords.horizontal).copy(
                lfluxland=True
            )

        self._step_function = self._generate_step_function(jitted=self.jitted)

        seed_grid_field_defaults(
            component,
            _jax_gcm_runtime.jax_gcm_default_field_names(
                include_total_surface_temperature=False,
            ),
            context,
            overrides={
                "sea_surface_temperature": (
                    _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE
                ),
            },
        )

        self.output_adapter.reset()

        if self.do_spinup and "OCN" in context.run_sequence:

            def spinup_step(step_number: int) -> None:
                _ = step_number
                _new_state, _predictions = self._step_function(
                    self._state,
                    self.forcing,
                )
                self._state = _new_state
                self.output_adapter.accumulate(
                    _jax_gcm_output.jax_gcm_prediction_output_variables(
                        _predictions,
                        coords=self.model.coords,
                        physics_module=getattr(self.model, "physics", None),
                    ),
                    summation_dim=_jax_gcm_output.JAX_GCM_TIME_DIM,
                )

            run_logged_spinup(
                steps=self.spinup_steps,
                logger=context.logger,
                intro_message=f" Performing JCM spinup for {self.spinup_time} day(s)...",
                step_message=lambda step, total: f" JCM spinup step {step} / {total}",
                step=spinup_step,
            )


__all__ = [
    "JAXGCMSetupState",
    "JCMState",
]
