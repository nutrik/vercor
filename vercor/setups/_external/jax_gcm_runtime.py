from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from vercor.components import (
    Component,
    PrefillContext,
    PrefillResult,
    StepContext,
    StepResult,
    ValidationContext,
)
from vercor.dtypes import DTypePolicy, as_jax_real_array, jax_full, jax_zeros
from vercor.exceptions import ComponentError, CouplerError
from vercor.field_layout import validate_canonical_grid_field_shape
from vercor._pytree import PyTreeNodeMixin
from vercor.physics import PhysicalConstants
from vercor.setups._external._jax_gcm_pytree import tree_as_runtime_dtype
import vercor.setups._external.jax_gcm_fields as _jax_gcm_fields

if TYPE_CHECKING:
    from vercor.setups._external.jax_gcm_state import JAXGCMSetupState

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
        *_jax_gcm_fields.JAXGCM_INPUT_GRID_FIELD_NAMES,
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


def create_jax_gcm_runtime_payload(
    state: "JAXGCMSetupState",
    component: Component | None = None,
) -> JAXGCMRuntimePayload:
    """Return immutable JCM state and forcing for runtime execution."""

    _ = component
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

    jcm_state = tree_as_runtime_dtype(state._state, state._dtype_policy)
    forcing = tree_as_runtime_dtype(state.forcing, state._dtype_policy)
    return JAXGCMRuntimePayload(
        jcm_state=jcm_state,
        forcing=forcing,
    )


def prefill_jax_gcm_runtime_fields(
    state: "JAXGCMSetupState",
    component: Component,
    context: PrefillContext,
) -> PrefillResult:
    """Pre-seed JAXGCM output fields so scan carry structure is stable."""

    data = dict(context.fields)
    defaults = {
        name: 0.0
        for name in jax_gcm_default_field_names(
            include_total_surface_temperature=True,
        )
    }
    defaults["sea_surface_temperature"] = _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE
    for name, value in defaults.items():
        data.setdefault(
            name, jax_full(component.grid.shape, value, state._dtype_policy)
        )
    sigma_levels = jnp.asarray(state.sigma_levels)
    data.setdefault(
        "pressure",
        jax_zeros((sigma_levels.shape[0], *component.grid.shape), state._dtype_policy),
    )
    return PrefillResult(fields=data)


def validate_jax_gcm_runtime_state(
    state: "JAXGCMSetupState",
    component: Component,
    context: ValidationContext,
) -> None:
    """Validate JAXGCM runtime payload and pre-seeded output fields."""

    if not isinstance(context.payload, JAXGCMRuntimePayload):
        raise ComponentError(
            "JAXGCM runtime requires an initialized immutable runtime payload "
            f"for component '{component.name}'"
        )

    for field_name in _jax_gcm_fields.JAXGCM_REQUIRED_GRID_FIELD_NAMES:
        if field_name not in context.state.fields():
            raise CouplerError(
                "Runtime missing required data field "
                f"'{field_name}' for component '{component.name}'"
            )
        try:
            validate_canonical_grid_field_shape(
                field_name=field_name,
                value=context.state.field(field_name, scope="state"),
                grid_shape=component.grid.shape,
                owner_description="Runtime required data field",
                owner_name=component.name,
            )
        except ValueError as exc:
            raise CouplerError(str(exc)) from exc

    if "pressure" not in context.state.fields():
        raise CouplerError(
            "Runtime missing required data field "
            f"'pressure' for component '{component.name}'"
        )
    pressure_shape = jnp.asarray(context.state.field("pressure", scope="state")).shape
    sigma_levels = jnp.asarray(state.sigma_levels)
    expected_pressure_shape = (sigma_levels.shape[0], *component.grid.shape)
    if pressure_shape != expected_pressure_shape:
        raise CouplerError(
            "Runtime required data field 'pressure' "
            f"for component '{component.name}' has shape {pressure_shape}, "
            f"expected {expected_pressure_shape}"
        )


def _required_speedy_diagnostics(
    physics: Mapping[str, Any],
    *,
    component_name: str,
) -> tuple[Any, Any]:
    """Return required SPEEDY diagnostics from a JCM 2 mapping."""

    required = ("_surface_flux", "_shortwave_rad")
    missing = tuple(name for name in required if name not in physics)
    if missing:
        names = ", ".join(missing)
        raise ComponentError(
            f"JAXGCM component '{component_name}' is missing JCM diagnostics: {names}"
        )
    return physics["_surface_flux"], physics["_shortwave_rad"]


def step_jax_gcm_runtime(
    state: "JAXGCMSetupState",
    fields: Mapping[str, Any],
    payload: Any | None,
    constants: PhysicalConstants,
    dtype: DTypePolicy,
) -> tuple[StepResult, Any, Any]:
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
    ) = _jax_gcm_fields.cleanup_surface_temperature_fields(
        fields.get("land_surface_temperature"),
        fields.get("sea_surface_temperature"),
    )

    land_surface_temperature_forcing, sea_surface_temperature_forcing = (
        _jax_gcm_fields.prepare_surface_temperature_forcing(
            total_surface_temperature,
            as_jax_real_array(state.model.terrain.fmask, dtype).T,
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
    surface_flux, shortwave_rad = _required_speedy_diagnostics(
        jcm_state.physics,
        component_name=state.name,
    )

    mapped_fields = _jax_gcm_fields.map_jcm_output_fields(
        constants.latent_heat_of_vaporization,
        JCM_REFERENCE_PRESSURE,
        state.sigma_levels,
        constants.dry_air_molecular_weight,
        constants.universal_gas_constant,
        constants.reference_pressure,
        constants.dry_air_kappa,
        surface_flux.shf,
        surface_flux.evap,
        surface_flux.rlds,
        shortwave_rad.rsns,
        jcm_state.dynamics.normalized_surface_pressure,
        jcm_state.dynamics.u_wind,
        jcm_state.dynamics.v_wind,
        jcm_state.dynamics.temperature,
        jcm_state.dynamics.specific_humidity,
        dtype=dtype,
    )

    step_result = StepResult(
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
    state: "JAXGCMSetupState",
    *,
    step_result: StepResult,
    prediction: Any,
    applied_forcing: Any,
    context: StepContext,
) -> None:
    """Record host-side JAXGCM mirrors and optional period output."""

    logger = context.logger
    if isinstance(step_result.payload, JAXGCMRuntimePayload):
        state._state = step_result.payload.jcm_state
        state.forcing = applied_forcing
    _, _, _, cold_surface_cells = _jax_gcm_fields.cleanup_surface_temperature_fields(
        step_result.fields.get("land_surface_temperature"),
        step_result.fields.get("sea_surface_temperature"),
    )
    if logger is not None:
        logger.info(
            "Number of cells with (SST + SKT) less than 250.0 K: {}",
            jnp.sum(cold_surface_cells),
        )

    _ = prediction


def step_jax_gcm_component(
    state: "JAXGCMSetupState",
    fields: Mapping[str, Any],
    context: StepContext,
    payload: Any | None,
) -> StepResult:
    """Advance JAXGCM on immutable runtime state."""

    time = context.time
    logger = context.logger
    if logger is not None:
        logger.info(
            "Mean of SST: {}",
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
        context.constants,
        context.dtype,
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
