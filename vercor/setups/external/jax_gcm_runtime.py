from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from vercor.components import Component, ComponentStepContext, ComponentStepResult
from vercor.dtypes import as_jax_real_array, jax_zeros
from vercor.exceptions import ComponentError, CouplerError
from vercor.pytree import PyTreeNodeMixin
from vercor.pytree_utils import mean_leaf, stack_objects, unwrap_leading_dims
import vercor.setups.external.jax_gcm_fields as _jax_gcm_fields
from vercor.setups.external.jax_gcm_output import (
    should_write_period_output,
    write_jax_gcm_averages_output,
)
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentContract, RuntimeComponentState

JCM_REFERENCE_PRESSURE = 1.0e5


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JAXGCMRuntimePayload(PyTreeNodeMixin):
    """Immutable JAXGCM model state carried by runtime component state."""

    pytree_children = ("jcm_state", "forcing")

    jcm_state: Any
    forcing: Any


def jax_gcm_default_field_names(
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


def jax_gcm_default_fields() -> dict[str, float]:
    """Return scalar defaults for the JAXGCM runtime output contract."""

    defaults = {
        field_name: 0.0
        for field_name in jax_gcm_default_field_names(
            include_total_surface_temperature=True
        )
    }
    defaults["sea_surface_temperature"] = _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE
    return defaults


def create_jax_gcm_runtime_payload(state: Any) -> JAXGCMRuntimePayload:
    """Return immutable JCM state and forcing for runtime execution."""

    missing = [
        name
        for name in ("_state", "forcing", "_step_function")
        if not hasattr(state, name)
    ]
    if missing:
        missing_names = ", ".join(missing)
        raise ComponentError(
            "JAXGCM runtime requires component initialization before "
            f"state creation; missing {missing_names}"
        )

    return JAXGCMRuntimePayload(
        jcm_state=state._state,
        forcing=state.forcing,
    )


def prefill_jax_gcm_runtime_fields(
    state: Any,
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
            jax_gcm_default_field_names(
                include_total_surface_temperature=True,
            ),
            overrides={
                "sea_surface_temperature": (
                    _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE
                ),
            },
        ),
    )
    sigma_levels = jnp.asarray(state.sigma_levels)
    data.setdefault(
        "pressure",
        jax_zeros((sigma_levels.shape[0], *component.grid.shape), component.settings),
    )
    _ = incoming, outgoing, contract


def validate_jax_gcm_runtime_state(
    state: Any,
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
    sigma_levels = jnp.asarray(state.sigma_levels)
    expected_pressure_shape = (sigma_levels.shape[0], *component.grid.shape)
    if pressure_shape != expected_pressure_shape:
        raise CouplerError(
            "Runtime required data field 'pressure' "
            f"for component '{component.name}' has shape {pressure_shape}, "
            f"expected {expected_pressure_shape}"
        )


def step_jax_gcm_runtime(
    state: Any,
    fields: Mapping[str, Any],
    payload: Any | None,
    settings: Any,
) -> tuple[ComponentStepResult, Any, Any]:
    """Advance JAXGCM runtime state and return raw prediction details."""

    if not isinstance(payload, JAXGCMRuntimePayload):
        raise ComponentError(
            "JAXGCM runtime requires an initialized immutable runtime payload "
            f"for component '{state.name}'"
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
            as_jax_real_array(state.model.terrain.fmask, settings).T,
        )
    )
    applied_forcing = payload.forcing.copy(
        stl_am=land_surface_temperature_forcing.T,
        sea_surface_temperature=sea_surface_temperature_forcing.T,
    )
    jcm_state, prediction = state._step_function(
        payload.jcm_state,
        applied_forcing,
    )
    averaged_prediction = mean_leaf(
        unwrap_leading_dims(stack_objects([prediction])), axis=0
    )

    mapped_fields = _jax_gcm_fields._map_jcm_output_fields(
        settings.latvap,
        JCM_REFERENCE_PRESSURE,
        state.sigma_levels,
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


def record_jax_gcm_host_step(
    state: Any,
    *,
    step_result: ComponentStepResult,
    prediction: Any,
    applied_forcing: Any,
    context: ComponentStepContext,
) -> None:
    """Record host-side JAXGCM mirrors and optional period output."""

    logger = context.logger
    if isinstance(step_result.payload, JAXGCMRuntimePayload):
        state._state = step_result.payload.jcm_state
        state.forcing = applied_forcing
    state._predictions_list.append(prediction)

    _, _, _, cold_surface_cells = _jax_gcm_fields._cleanup_surface_temperature_fields(
        step_result.fields.get("land_surface_temperature"),
        step_result.fields.get("sea_surface_temperature"),
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
        output_frequency=state.output_frequency,
    ):
        date_time = time.strftime("%Y-%m-%d")
        write_jax_gcm_averages_output(
            state._predictions_list,
            output=f"jcm.averages.{date_time}.nc",
            logger=logger,
        )


def step_jax_gcm_component(
    state: Any,
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
    ) = step_jax_gcm_runtime(
        state,
        fields,
        payload,
        context.settings,
    )

    if time is not None:
        record_jax_gcm_host_step(
            state,
            step_result=step_result,
            prediction=prediction,
            applied_forcing=applied_forcing,
            context=context,
        )
    return step_result


__all__ = [
    "JAXGCMRuntimePayload",
    "create_jax_gcm_runtime_payload",
    "jax_gcm_default_field_names",
    "jax_gcm_default_fields",
    "prefill_jax_gcm_runtime_fields",
    "record_jax_gcm_host_step",
    "step_jax_gcm_component",
    "step_jax_gcm_runtime",
    "validate_jax_gcm_runtime_state",
]
