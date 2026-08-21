"""JAXGCM/JCM atmosphere component factory."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from vercor.components import (
    CallableComponent,
    Component,
    LifecycleHooks,
    ComponentSpec,
)
from vercor.output import OutputSpec
from vercor.setups.config import JAXGCMConfig

if TYPE_CHECKING:
    from dinosaur.coordinate_systems import CoordinateSystem as _CoordinateSystem
    from jcm.terrain import TerrainData as _TerrainData
else:
    _CoordinateSystem = Any
    _TerrainData = Any


def make_jax_gcm(
    coords: _CoordinateSystem,
    terrain: _TerrainData,
    *,
    config: JAXGCMConfig | None = None,
) -> Component:
    """Return a differentiable JAXGCM/JCM atmosphere component."""

    try:
        import jcm  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "The JAXGCM component requires the jcm package. Please install it with "
            "`pip install jcm`."
        ) from error

    import vercor.setups._external.jax_gcm_fields as _jax_gcm_fields
    import vercor.setups._external.jax_gcm_output as _jax_gcm_output
    import vercor.setups._external.jax_gcm_runtime as _jax_gcm_runtime
    from vercor.setups._external.jax_gcm_state import JAXGCMSetupState

    config = JAXGCMConfig() if config is None else config
    state = JAXGCMSetupState(
        coords=coords,
        terrain=terrain,
        name=config.name,
        custom_parameters=config.custom_parameters,
        model_timestep=config.model_timestep,
        save_interval=config.save_interval,
        spinup_time=config.spinup.duration,
        forcing_data=config.forcing_data,
        do_spinup=config.spinup.enabled,
        jitted=config.jitted,
    )
    initial_fields: dict[str, object] = dict(_jax_gcm_runtime.jax_gcm_default_fields())
    initial_fields["pressure"] = jnp.zeros(
        (state.sigma_levels.shape[0], *state.grid.shape)
    )
    component = CallableComponent(
        config.name,
        state.grid,
        partial(_jax_gcm_runtime.step_jax_gcm_component, state),
        spec=ComponentSpec(
            inputs=_jax_gcm_fields.JAXGCM_INPUT_GRID_FIELD_NAMES,
            outputs=(
                "land_surface_temperature",
                "sea_surface_temperature",
                "total_surface_temperature",
                *_jax_gcm_fields.JAXGCM_OUTPUT_GRID_FIELD_NAMES,
                "pressure",
            ),
            initial_fields=initial_fields,
            lifecycle=LifecycleHooks(
                setup=state.setup,
                prefill=partial(
                    _jax_gcm_runtime.prefill_jax_gcm_runtime_fields,
                    state,
                ),
                validate=partial(
                    _jax_gcm_runtime.validate_jax_gcm_runtime_state,
                    state,
                ),
            ),
            output=OutputSpec(
                provider=(
                    _jax_gcm_output.jax_gcm_output_provider(state)
                    if config.output.provider is None
                    else config.output.provider
                ),
                snapshot_writer=config.output.snapshot_writer
                or partial(_jax_gcm_output.write_jax_gcm_snapshot_output, state),
                period=config.output.period,
            ),
        ),
    )
    return component


__all__ = [
    "make_jax_gcm",
]
