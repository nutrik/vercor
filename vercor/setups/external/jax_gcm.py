from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp
import tree_math

from dinosaur import primitive_equations
from dinosaur.coordinate_systems import CoordinateSystem

from jcm.constants import p0
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
from vercor.exceptions import ComponentError, CouplerError
from vercor.setups._time_helpers import (
    assign_model_timestep_alignment,
    run_logged_spinup,
    seed_grid_field_defaults,
)
from vercor.setups.external.jax_gcm_tools import (
    change_jcm_parameter_values,
)
from vercor.setups.external.jax_gcm_output import (
    should_write_period_output,
    write_jax_gcm_averages_output,
)
import vercor.setups.external.jax_gcm_fields as _jax_gcm_fields
from vercor.pytree_utils import asfloat, mean_leaf, stack_objects, unwrap_leading_dims
from vercor.dtypes import (
    as_jax_real_array,
    jax_ones,
    jax_zeros,
)
from vercor.grid import RectilinearGrid
from vercor.pytree import PyTreeNodeMixin
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentContract, RuntimeComponentState


try:
    import jcm  # noqa: F401
except ImportError:
    raise ImportError(
        "The JAXGCM component requires the jcm package. Please install it with `pip install jcm`."
    )


def _jax_gcm_default_field_names(
    *,
    include_total_surface_temperature: bool,
) -> tuple[str, ...]:
    """Return JAXGCM grid-field default names in stable insertion order."""

    fields = (
        *_jax_gcm_fields.JAXGCM_OUTPUT_GRID_FIELD_NAMES,
        "land_surface_temperature",
        "sea_surface_temperature",
    )
    if include_total_surface_temperature:
        return (*fields, "total_surface_temperature")
    return fields


@tree_math.struct
@dataclass
class JCMState:
    prog: PhysicsState
    phydata: Any
    metadata: primitive_equations.State


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JAXGCMRuntimePayload(PyTreeNodeMixin):
    """Immutable JAXGCM model state carried by runtime component state."""

    pytree_children = ("jcm_state", "forcing")

    jcm_state: Any
    forcing: Any


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
            _jax_gcm_default_field_names(
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

    def create_runtime_payload(self) -> JAXGCMRuntimePayload:
        """Return immutable JCM state and forcing for runtime execution."""

        missing = [
            name
            for name in ("_state", "forcing", "_step_function")
            if not hasattr(self, name)
        ]
        if missing:
            missing_names = ", ".join(missing)
            raise ComponentError(
                "JAXGCM runtime requires component initialization before "
                f"state creation; missing {missing_names}"
            )

        return JAXGCMRuntimePayload(
            jcm_state=self._state,
            forcing=self.forcing,
        )

    def prefill_runtime_state_fields(
        self,
        component: Component,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
        contract: "RuntimeComponentContract",
    ) -> None:
        """Pre-seed JAXGCM output fields so scan carry structure is stable."""

        component.prefill_runtime_fields(
            data,
            default_fields=component.grid_field_defaults(
                _jax_gcm_default_field_names(
                    include_total_surface_temperature=True,
                ),
                overrides={
                    "sea_surface_temperature": (
                        _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE
                    ),
                },
            ),
        )
        sigma_levels = jnp.asarray(self.sigma_levels)
        data.setdefault(
            "pressure",
            jax_zeros(
                (sigma_levels.shape[0], *component.grid.shape), component.settings
            ),
        )
        _ = incoming, outgoing, contract

    def validate_runtime_state(
        self,
        component: Component,
        component_state: "RuntimeComponentState",
        contract: "RuntimeComponentContract",
    ) -> None:
        """Validate JAXGCM runtime payload and pre-seeded output fields."""

        _ = contract
        if not isinstance(component_state.runtime_payload, JAXGCMRuntimePayload):
            raise ComponentError(
                "JAXGCM runtime requires an initialized immutable runtime payload "
                f"for component '{component.name}'"
            )

        component.require_runtime_fields(
            component_state, *_jax_gcm_fields.JAXGCM_REQUIRED_GRID_FIELD_NAMES
        )

        if "pressure" not in component_state.data:
            raise CouplerError(
                "Runtime missing required data field "
                f"'pressure' for component '{component.name}'"
            )
        pressure_shape = jnp.asarray(component_state.data.get("pressure")).shape
        sigma_levels = jnp.asarray(self.sigma_levels)
        expected_pressure_shape = (sigma_levels.shape[0], *component.grid.shape)
        if pressure_shape != expected_pressure_shape:
            raise CouplerError(
                "Runtime required data field 'pressure' "
                f"for component '{component.name}' has shape {pressure_shape}, "
                f"expected {expected_pressure_shape}"
            )

    def _step_jax_gcm_component_state(
        self,
        fields: Mapping[str, Any],
        payload: Any | None,
        settings: Any,
    ) -> tuple[ComponentStepResult, Predictions, Any]:
        """Advance JAXGCM runtime state and return the raw prediction."""

        if not isinstance(payload, JAXGCMRuntimePayload):
            raise ComponentError(
                "JAXGCM runtime requires an initialized immutable runtime payload "
                f"for component '{self.name}'"
            )

        (
            land_surface_temperature,
            sea_surface_temperature,
            total_surface_temperature,
            _,
        ) = _jax_gcm_fields._cleanup_surface_temperature_fields(
            fields.get("land_surface_temperature"),
            fields.get("sea_surface_temperature"),
        )

        land_surface_temperature_forcing, sea_surface_temperature_forcing = (
            _jax_gcm_fields._prepare_surface_temperature_forcing(
                total_surface_temperature,
                as_jax_real_array(self.model.terrain.fmask, settings).T,
            )
        )
        applied_forcing = payload.forcing.copy(
            stl_am=land_surface_temperature_forcing.T,
            sea_surface_temperature=sea_surface_temperature_forcing.T,
        )
        jcm_state, prediction = self._step_function(
            payload.jcm_state,
            applied_forcing,
        )
        averaged_prediction = mean_leaf(
            unwrap_leading_dims(stack_objects([prediction])), axis=0
        )

        mapped_fields = _jax_gcm_fields._map_jcm_output_fields(
            settings.latvap,
            p0,
            self.sigma_levels,
            settings.mwdair,
            settings.rgas,
            settings.p0,
            settings.cappa,
            averaged_prediction.physics.surface_flux.shf,
            averaged_prediction.physics.surface_flux.evap,
            averaged_prediction.physics.surface_flux.rlds,
            averaged_prediction.physics.shortwave_rad.rsns,
            averaged_prediction.dynamics.normalized_surface_pressure,
            averaged_prediction.dynamics.u_wind,
            averaged_prediction.dynamics.v_wind,
            averaged_prediction.dynamics.temperature,
            averaged_prediction.dynamics.specific_humidity,
        )

        step_result = ComponentStepResult(
            fields={
                "land_surface_temperature": land_surface_temperature,
                "sea_surface_temperature": sea_surface_temperature,
                "total_surface_temperature": total_surface_temperature,
                **mapped_fields,
            },
            payload=JAXGCMRuntimePayload(
                jcm_state=jcm_state,
                forcing=payload.forcing,
            ),
        )

        return (
            step_result,
            prediction,
            applied_forcing,
        )

    def step(
        self,
        fields: Mapping[str, Any],
        context: ComponentStepContext,
        payload: Any | None,
    ) -> ComponentStepResult:
        """Advance JAXGCM on immutable runtime state."""

        time = context.time
        logger = context.logger
        if logger is not None:
            logger.info(
                " Mean of SST: {}",
                jnp.nanmean(jnp.asarray(fields.get("sea_surface_temperature"))),
            )

        (
            step_result,
            prediction,
            applied_forcing,
        ) = self._step_jax_gcm_component_state(
            fields,
            payload,
            context.settings,
        )

        if time is None:
            return step_result

        self._record_jax_gcm_host_step(
            step_result=step_result,
            prediction=prediction,
            applied_forcing=applied_forcing,
            context=context,
        )
        return step_result

    def _record_jax_gcm_host_step(
        self,
        *,
        step_result: ComponentStepResult,
        prediction: Predictions,
        applied_forcing: Any,
        context: ComponentStepContext,
    ) -> None:
        """Record host-side JAXGCM mirrors and optional period output."""

        logger = context.logger
        if isinstance(step_result.payload, JAXGCMRuntimePayload):
            self._state = step_result.payload.jcm_state
            self.forcing = applied_forcing
        self._predictions_list.append(prediction)

        _, _, _, cold_surface_cells = (
            _jax_gcm_fields._cleanup_surface_temperature_fields(
                step_result.fields.get("land_surface_temperature"),
                step_result.fields.get("sea_surface_temperature"),
            )
        )
        if logger is not None:
            logger.info(
                " Number of cells with (SST + SKT) less than 250.0 K: {}",
                jnp.sum(cold_surface_cells),
            )

        time = context.time
        if time is not None and should_write_period_output(
            time=time,
            dt=timedelta(seconds=context.dt_seconds),
            output_frequency=self.output_frequency,
        ):
            date_time = time.strftime("%Y-%m-%d")
            write_jax_gcm_averages_output(
                self._predictions_list,
                output=f"jcm.averages.{date_time}.nc",
                logger=logger,
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
    default_fields = {
        field_name: 0.0
        for field_name in _jax_gcm_default_field_names(
            include_total_surface_temperature=True
        )
    }
    default_fields["sea_surface_temperature"] = (
        _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE
    )
    component = differentiable_component(
        name=name,
        grid=state.grid,
        step=state.step,
        inputs=("land_surface_temperature", "sea_surface_temperature"),
        outputs=(
            "land_surface_temperature",
            "sea_surface_temperature",
            "total_surface_temperature",
            *_jax_gcm_fields.JAXGCM_OUTPUT_GRID_FIELD_NAMES,
            "pressure",
        ),
        default_fields=default_fields,
        initialize=state.initialize,
        create_runtime_payload=lambda component: state.create_runtime_payload(),
        prefill_runtime_state_fields=state.prefill_runtime_state_fields,
        validate_runtime_state=state.validate_runtime_state,
    )
    return component
