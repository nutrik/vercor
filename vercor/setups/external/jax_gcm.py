from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import tree_math

from dinosaur import primitive_equations
from dinosaur.coordinate_systems import CoordinateSystem

from jcm.forcing import default_forcing
from jcm.model import ForcingData, Model, Predictions
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics_interface import (
    PhysicsState,
    TerrainData,
    dynamics_state_to_physics_state,
)

from vercor.components import (
    Component,
    ComponentSetupContext,
    ComponentStepContext,
    ComponentStepResult,
    differentiable_component,
)
from vercor.setups._time_helpers import (
    assign_model_timestep_alignment,
    run_logged_spinup,
    seed_grid_field_defaults,
)
from vercor.setups.external.jax_gcm_tools import (
    change_jcm_parameter_values,
)
import vercor.setups.external.jax_gcm_fields as _jax_gcm_fields
import vercor.setups.external.jax_gcm_runtime as _jax_gcm_runtime
from vercor.pytree_utils import asfloat, mean_leaf
from vercor.dtypes import (
    as_jax_real_array,
    jax_ones,
)
from vercor.grid import RectilinearGrid
from vercor.types import RuntimeArray

try:
    import jcm  # noqa: F401
except ImportError:
    raise ImportError(
        "The JAXGCM component requires the jcm package. Please install it with "
        "`pip install jcm`."
    )


@tree_math.struct
@dataclass
class JCMState:
    prog: PhysicsState
    phydata: Any
    metadata: primitive_equations.State


class _JAXGCMState:
    """JCM Wrapper"""

    _predictions_list: list[Predictions]
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
        custom_parameters: Optional[dict[str, float]] = None,
        model_timestep: timedelta = timedelta(minutes=30),
        save_interval: timedelta = timedelta(days=1),
        spinup_time: timedelta = timedelta(days=2),
        forcing_data: Optional[ForcingData] = None,
        # Output frequency in days for saving JCM predictions.
        output_frequency: Optional[str] = None,
        do_spinup: bool = False,
        jitted: bool = True,
    ) -> None:

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
            ).transpose(),  # This is used for interpolation, which all points are valid
        )

        self.sigma_levels: RuntimeArray = self.model.coords.vertical.centers

        self.name = name
        self.grid = grid
        self.settings: Any | None = None

    def _generate_step_function(
        self, jitted: bool = True
    ) -> Callable[[JCMState, ForcingData], tuple[JCMState, Predictions]]:
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

            # phydata is a stacked object, so I take the mean here.
            # However, this action will be done by jcm in the new jcm PR.
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

        self._predictions_list = []

        if self.do_spinup and "OCN" in context.run_sequence.order:

            def spinup_step(step_number: int) -> None:
                _ = step_number
                _new_state, _predictions = self._step_function(
                    self._state,
                    self.forcing,
                )
                self._state = _new_state
                self._predictions_list.append(_predictions)

            run_logged_spinup(
                steps=self.spinup_steps,
                logger=context.logger,
                intro_message=f" Performing JCM spinup for {self.spinup_time} day(s)...",
                step_message=lambda step, total: f" JCM spinup step {step} / {total}",
                step=spinup_step,
            )


def _step_jax_gcm_runtime_callback(
    state: _JAXGCMState,
    fields: Mapping[str, Any],
    context: ComponentStepContext,
    payload: Any | None,
) -> ComponentStepResult:
    """Delegate callable component stepping to the JAXGCM runtime owner."""

    return _jax_gcm_runtime.step_jax_gcm_component(
        state,
        fields,
        context,
        payload,
    )


def _create_jax_gcm_runtime_payload_callback(
    state: _JAXGCMState,
    component: Component,
) -> _jax_gcm_runtime.JAXGCMRuntimePayload:
    """Delegate runtime-payload creation to the JAXGCM runtime owner."""

    _ = component
    return _jax_gcm_runtime.create_jax_gcm_runtime_payload(state)


def _prefill_jax_gcm_runtime_fields_callback(
    state: _JAXGCMState,
    component: Component,
    data: dict[str, RuntimeArray],
    incoming: dict[str, RuntimeArray],
    outgoing: dict[str, RuntimeArray],
    contract: Any,
) -> None:
    """Delegate runtime field prefill to the JAXGCM runtime owner."""

    _jax_gcm_runtime.prefill_jax_gcm_runtime_fields(
        state,
        component,
        data,
        incoming,
        outgoing,
        contract,
    )


def _validate_jax_gcm_runtime_state_callback(
    state: _JAXGCMState,
    component: Component,
    component_state: Any,
    contract: Any,
) -> None:
    """Delegate runtime-state validation to the JAXGCM runtime owner."""

    _jax_gcm_runtime.validate_jax_gcm_runtime_state(
        state,
        component,
        component_state,
        contract,
    )


def make_jax_gcm(
    coords: CoordinateSystem,
    terrain: TerrainData,
    name: str = "ATM",
    custom_parameters: Optional[dict[str, float]] = None,
    model_timestep: timedelta = timedelta(minutes=30),
    save_interval: timedelta = timedelta(days=1),
    spinup_time: timedelta = timedelta(days=2),
    forcing_data: Optional[ForcingData] = None,
    output_frequency: Optional[str] = None,
    do_spinup: bool = False,
    jitted: bool = True,
) -> Component:
    """Return a differentiable JAXGCM/JCM atmosphere component."""

    state = _JAXGCMState(
        coords=coords,
        terrain=terrain,
        name=name,
        custom_parameters=custom_parameters,
        model_timestep=model_timestep,
        save_interval=save_interval,
        spinup_time=spinup_time,
        forcing_data=forcing_data,
        output_frequency=output_frequency,
        do_spinup=do_spinup,
        jitted=jitted,
    )
    component = differentiable_component(
        name=name,
        grid=state.grid,
        step=partial(_step_jax_gcm_runtime_callback, state),
        inputs=("land_surface_temperature", "sea_surface_temperature"),
        outputs=(
            "land_surface_temperature",
            "sea_surface_temperature",
            "total_surface_temperature",
            *_jax_gcm_fields.JAXGCM_OUTPUT_GRID_FIELD_NAMES,
            "pressure",
        ),
        default_fields=_jax_gcm_runtime.jax_gcm_default_fields(),
        initialize=state.initialize,
        create_runtime_payload=partial(
            _create_jax_gcm_runtime_payload_callback,
            state,
        ),
        prefill_runtime_state_fields=partial(
            _prefill_jax_gcm_runtime_fields_callback,
            state,
        ),
        validate_runtime_state=partial(
            _validate_jax_gcm_runtime_state_callback,
            state,
        ),
    )
    return component
