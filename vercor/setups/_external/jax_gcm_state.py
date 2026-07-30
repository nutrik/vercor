"""JAXGCM setup-state ownership and lifecycle callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    tree_as_real_dtype,
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
    """JCM prognostic, physics, and primitive-equation state bundle."""

    prog: PhysicsState
    phydata: Any
    metadata: primitive_equations.State


class JAXGCMSetupState:
    """Mutable setup-time owner for a JAXGCM/JCM atmosphere adapter."""

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
        self._dtype_policy = DTypePolicy.from_jax_config()

    def _generate_step_function(
        self, jitted: bool = True
    ) -> Callable[[JCMState, ForcingData], tuple[JCMState, Predictions]]:
        """Return the model step function, optionally JIT compiled."""

        def step_function(
            state: JCMState, forcing: ForcingData
        ) -> tuple[JCMState, Predictions]:
            precision_policy = self._dtype_policy
            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state=state.metadata,
                save_interval=self.save_interval / timedelta(days=1),
                total_time=self.coupling_timestep / timedelta(days=1),
                forcing=forcing,
            )

            # JCM currently returns a stacked object; reduce to one runtime state.
            return (
                JCMState(
                    prog=tree_as_real_dtype(
                        tree_mean(predictions.dynamics, axis=0),
                        precision_policy,
                    ),
                    phydata=tree_as_real_dtype(
                        tree_mean(predictions.physics, axis=0),
                        precision_policy,
                    ),
                    metadata=new_atm_modal_state,
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

        _modal_state = self.model._prepare_initial_modal_state()
        speedy_coords = getattr(
            getattr(self.model, "physics", None),
            "cached_coords",
            None,
        )
        physics_data_kwargs = (
            {"speedy_coords": speedy_coords} if speedy_coords is not None else {}
        )
        self._state = JCMState(
            metadata=_modal_state,
            phydata=tree_as_real_dtype(
                PhysicsData.zeros(
                    self.model.coords.horizontal.nodal_shape,
                    self.model.coords.vertical.layers,
                    **physics_data_kwargs,
                ),
                self._dtype_policy,
            ),
            prog=dynamics_state_to_physics_state(_modal_state, self.model.primitive),
        )

        if self.forcing_data is not None:
            self.forcing = self.forcing_data
        else:
            self.forcing = default_forcing(self.model.coords.horizontal).copy(
                lfluxland=True
            )

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
