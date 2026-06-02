"""JAXGCM/JCM atmosphere component factory."""

from __future__ import annotations

from datetime import timedelta
from functools import partial

from dinosaur.coordinate_systems import CoordinateSystem
from jcm.model import ForcingData
from jcm.physics_interface import TerrainData

from vercor.components import Component, differentiable_component
import vercor.setups.external.jax_gcm_fields as _jax_gcm_fields
import vercor.setups.external.jax_gcm_runtime as _jax_gcm_runtime
import vercor.setups.external.jax_gcm_state as _jax_gcm_state
from vercor.setups.external.jax_gcm_state import JAXGCMSetupState

try:
    import jcm  # noqa: F401
except ImportError:
    raise ImportError(
        "The JAXGCM component requires the jcm package. Please install it with "
        "`pip install jcm`."
    )


def make_jax_gcm(
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
) -> Component:
    """Return a differentiable JAXGCM/JCM atmosphere component."""

    state = JAXGCMSetupState(
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
        step=partial(_jax_gcm_state.step_jax_gcm_runtime_callback, state),
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
            _jax_gcm_state.create_jax_gcm_runtime_payload_callback,
            state,
        ),
        prefill_runtime_state_fields=partial(
            _jax_gcm_state.prefill_jax_gcm_runtime_fields_callback,
            state,
        ),
        validate_runtime_state=partial(
            _jax_gcm_state.validate_jax_gcm_runtime_state_callback,
            state,
        ),
    )
    return component


__all__ = [
    "make_jax_gcm",
]
