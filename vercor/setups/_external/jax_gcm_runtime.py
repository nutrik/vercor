from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

from vercor._numerical_safety import require_active_finite
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
    from vercor.setups._external.jax_gcm_state import JCMAtmosphereSetupState

JCM_REFERENCE_PRESSURE = 1.0e5
JAXGCM_PHYSICAL_FIELD_DEFAULTS = {
    "land_surface_temperature": _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE,
    "air_temperature": _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE,
    "air_potential_temperature": _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE,
    "air_density": _jax_gcm_fields.REFERENCE_AIR_DENSITY,
    "model_level_height": _jax_gcm_fields.REFERENCE_MODEL_LEVEL_HEIGHT,
}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JCMAtmosphereRuntimePayload(PyTreeNodeMixin):
    """Immutable JAXGCM model state carried by runtime component state."""

    pytree_children = ("jcm_state", "forcing")

    jcm_state: Any
    forcing: Any


def _require_finite_numeric_tree(tree: Any, *, owner: str) -> None:
    """Reject non-finite inexact leaves with their native PyTree paths."""

    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, value in path_leaves:
        try:
            array = jnp.asarray(value)
        except (TypeError, ValueError):
            continue
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            continue
        leaf_path = jax.tree_util.keystr(path)
        require_active_finite(
            array,
            active_mask=None,
            owner=f"{owner}{leaf_path or ' root'}",
        )


def _require_finite_jax_gcm_state(
    jcm_state: Any,
    *,
    component_name: str,
    stage: str,
) -> None:
    """Validate every named numerical section of one native JCM carry."""

    for section in ("dynamics", "physics", "dycore_state", "physics_carry"):
        _require_finite_numeric_tree(
            getattr(jcm_state, section),
            owner=(
                f"JAXGCM component '{component_name}' {stage} " f"jcm_state.{section}"
            ),
        )


def _require_finite_jax_gcm_payload(
    payload: JCMAtmosphereRuntimePayload,
    *,
    component_name: str,
    stage: str,
) -> None:
    """Validate native state and forcing before payload storage or reuse."""

    _require_finite_jax_gcm_state(
        payload.jcm_state,
        component_name=component_name,
        stage=stage,
    )
    _require_finite_numeric_tree(
        payload.forcing,
        owner=f"JAXGCM component '{component_name}' {stage} forcing",
    )


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


def jax_gcm_default_fields(
    *,
    include_total_surface_temperature: bool = True,
) -> dict[str, float]:
    """Return scalar defaults for the JAXGCM runtime output contract."""

    defaults = {
        field_name: 0.0
        for field_name in jax_gcm_default_field_names(
            include_total_surface_temperature=include_total_surface_temperature
        )
    }
    defaults.update(JAXGCM_PHYSICAL_FIELD_DEFAULTS)
    defaults["sea_surface_temperature"] = _jax_gcm_fields.REFERENCE_SURFACE_TEMPERATURE
    return defaults


def create_jax_gcm_runtime_payload(
    state: "JCMAtmosphereSetupState",
    component: Component | None = None,
) -> JCMAtmosphereRuntimePayload:
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
    payload = JCMAtmosphereRuntimePayload(
        jcm_state=jcm_state,
        forcing=forcing,
    )
    _require_finite_jax_gcm_payload(
        payload,
        component_name=state.name,
        stage="setup output",
    )
    return payload


def prefill_jax_gcm_runtime_fields(
    state: "JCMAtmosphereSetupState",
    component: Component,
    context: PrefillContext,
) -> PrefillResult:
    """Pre-seed JAXGCM output fields so scan carry structure is stable."""

    data = dict(context.fields)
    for name, value in jax_gcm_default_fields().items():
        data.setdefault(
            name, jax_full(component.grid.shape, value, state._dtype_policy)
        )
    if state.seaice_coupling:
        for name in (
            *_jax_gcm_fields.JAXGCM_COUPLED_INPUT_GRID_FIELD_NAMES,
            *_jax_gcm_fields.JAXGCM_COUPLED_OUTPUT_GRID_FIELD_NAMES,
        ):
            data.setdefault(name, jax_zeros(component.grid.shape, state._dtype_policy))
    sigma_levels = jnp.asarray(state.sigma_levels)
    data.setdefault(
        "air_pressure_3d",
        jax_zeros((sigma_levels.shape[0], *component.grid.shape), state._dtype_policy),
    )
    return PrefillResult(fields=data)


def validate_jax_gcm_runtime_state(
    state: "JCMAtmosphereSetupState",
    component: Component,
    context: ValidationContext,
) -> None:
    """Validate JAXGCM runtime payload and pre-seeded output fields."""

    if not isinstance(context.payload, JCMAtmosphereRuntimePayload):
        raise ComponentError(
            "JAXGCM runtime requires an initialized immutable runtime payload "
            f"for component '{component.name}'"
        )

    required: tuple[str, ...] = _jax_gcm_fields.JAXGCM_REQUIRED_GRID_FIELD_NAMES
    if state.seaice_coupling:
        required = (
            *required,
            *_jax_gcm_fields.JAXGCM_COUPLED_INPUT_GRID_FIELD_NAMES,
            *_jax_gcm_fields.JAXGCM_COUPLED_OUTPUT_GRID_FIELD_NAMES,
        )
    for field_name in required:
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

    if "air_pressure_3d" not in context.state.fields():
        raise CouplerError(
            "Runtime missing required data field "
            f"'air_pressure_3d' for component '{component.name}'"
        )
    pressure_shape = jnp.asarray(
        context.state.field("air_pressure_3d", scope="state")
    ).shape
    sigma_levels = jnp.asarray(state.sigma_levels)
    expected_pressure_shape = (sigma_levels.shape[0], *component.grid.shape)
    if pressure_shape != expected_pressure_shape:
        raise CouplerError(
            "Runtime required data field 'air_pressure_3d' "
            f"for component '{component.name}' has shape {pressure_shape}, "
            f"expected {expected_pressure_shape}"
        )


def _required_speedy_diagnostics(
    physics: Mapping[str, Any],
    *,
    component_name: str,
) -> tuple[Any, Any, Any, Any]:
    """Return required SPEEDY diagnostics from a JCM 2 mapping."""

    required = ("_surface_flux", "_shortwave_rad", "_condensation", "_convection")
    missing = tuple(name for name in required if name not in physics)
    if missing:
        names = ", ".join(missing)
        raise ComponentError(
            f"JAXGCM component '{component_name}' is missing JCM diagnostics: {names}"
        )
    return (
        physics["_surface_flux"],
        physics["_shortwave_rad"],
        physics["_condensation"],
        physics["_convection"],
    )


def _map_jax_gcm_state_fields(
    state: "JCMAtmosphereSetupState",
    jcm_state: Any,
    constants: PhysicalConstants,
    dtype: DTypePolicy,
) -> dict[str, jax.Array]:
    """Map one diagnosed native snapshot with shared units and channel selection."""
    surface_flux, shortwave_rad, condensation, convection = (
        _required_speedy_diagnostics(
            jcm_state.physics,
            component_name=state.name,
        )
    )

    return cast(
        dict[str, jax.Array],
        _jax_gcm_fields.map_jcm_output_fields(
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
            downward_shortwave_radiation_flux=shortwave_rad.rsds,
            open_water_albedo=state.open_water_albedo,
            large_scale_precipitation=condensation.precls,
            convective_precipitation=convection.precnv,
            near_surface_air_temperature=surface_flux.t0,
            snow_temperature=constants.precipitation_snow_temperature,
            rain_temperature=constants.precipitation_rain_temperature,
            dtype=dtype,
        ),
    )


def diagnose_jax_gcm_initial_fields(
    state: "JCMAtmosphereSetupState",
    payload: JCMAtmosphereRuntimePayload,
    constants: PhysicalConstants,
    dtype: DTypePolicy,
) -> tuple[dict[str, Any], JCMAtmosphereRuntimePayload]:
    """Evaluate current native physics without advancing dycore or cross-step carry.

    The diagnostic snapshot uses the original climatological surface until the
    first scheduled atmosphere step receives coupled component fields.
    """
    current = state.model.dycore.to_physics_state(payload.jcm_state.dycore_state)
    from jcm.date import DateData

    date = DateData.set_date(
        payload.jcm_state.time,
        model_step=payload.jcm_state.step,
        dt_seconds=state.model_timestep.total_seconds(),
    )
    selected = payload.forcing.select(date)
    _, diagnostics = state.model.physics.compute_tendencies(
        current,
        selected,
        state.model.terrain,
        prev_physics_data=payload.jcm_state.physics_carry,
    )
    snapshot = replace(
        payload.jcm_state,
        dynamics=tree_as_runtime_dtype(current, dtype),
        physics=tree_as_runtime_dtype(diagnostics, dtype),
    )
    fields: dict[str, Any] = _map_jax_gcm_state_fields(
        state, snapshot, constants, dtype
    )
    land = as_jax_real_array(state.model.terrain.fmask, dtype).T
    land_temperature = as_jax_real_array(selected.stl_am, dtype).T
    sea_temperature = as_jax_real_array(selected.sea_surface_temperature, dtype).T
    fields.update(
        {
            "land_surface_temperature": land_temperature,
            "sea_surface_temperature": sea_temperature,
            "soil_moisture": as_jax_real_array(selected.soilw_am, dtype).T,
            "total_surface_temperature": land * land_temperature
            + (1 - land) * sea_temperature,
            "marine_surface_albedo": as_jax_real_array(
                diagnostics["_mod_radcon"].alb_s, dtype
            ).T,
            "marine_ice_fraction": as_jax_real_array(selected.sice_am, dtype).T,
        }
    )
    return fields, JCMAtmosphereRuntimePayload(snapshot, payload.forcing)


def step_jax_gcm_runtime(
    state: "JCMAtmosphereSetupState",
    fields: Mapping[str, Any],
    payload: Any | None,
    constants: PhysicalConstants,
    dtype: DTypePolicy,
) -> StepResult:
    """Advance JAXGCM runtime state and replace its payload functionally."""

    if not isinstance(payload, JCMAtmosphereRuntimePayload):
        raise ComponentError(
            "JAXGCM runtime requires an initialized immutable runtime payload "
            f"for component '{state.name}'"
        )
    _require_finite_jax_gcm_payload(
        payload,
        component_name=state.name,
        stage="step input",
    )

    surface_fields: dict[str, Any]
    from vercor.setups._external.jax_gcm_surface import blend_jcm_forcing_field

    land_fraction = as_jax_real_array(state.model.terrain.fmask, dtype).T
    ocean_fraction, prescribed_area, _ = _jax_gcm_fields._marine_surface_fractions(
        land_fraction,
        (
            0.0
            if state.prescribed_marine_fraction is None
            else state.prescribed_marine_fraction
        ),
        dtype=dtype,
    )
    prescribed_weight = prescribed_area / jnp.where(
        ocean_fraction > 0, ocean_fraction, 1
    )
    if state.seaice_coupling:
        from vercor.setups._external.jax_gcm_surface import CoupledJCMForcing

        composed = _jax_gcm_fields.prepare_coupled_surface_forcing(
            {
                name: fields[name]
                for name in _jax_gcm_fields.JAXGCM_COUPLED_INPUT_GRID_FIELD_NAMES
            },
            land_fraction,
            state.open_water_albedo,
            prescribed_marine_fraction=prescribed_area,
            dtype=dtype,
        )
        applied_forcing = CoupledJCMForcing(
            payload.forcing.copy(
                stl_am=composed.land_surface_temperature.T,
                sea_surface_temperature=blend_jcm_forcing_field(
                    payload.forcing.sea_surface_temperature,
                    composed.sea_surface_temperature.T,
                    prescribed_weight.T,
                ),
                soilw_am=composed.soil_moisture.T,
                sice_am=blend_jcm_forcing_field(
                    payload.forcing.sice_am,
                    composed.marine_ice_fraction.T,
                    prescribed_weight.T,
                ),
            ),
            composed.marine_surface_albedo.T,
            native_background=payload.forcing,
            prescribed_marine_fraction=prescribed_weight.T,
        )
        surface_fields = {
            name: getattr(composed, name)
            for name in (
                "land_surface_temperature",
                "sea_surface_temperature",
                "soil_moisture",
                "total_surface_temperature",
                "marine_surface_albedo",
                "marine_ice_fraction",
            )
        }
    else:
        land_surface_temperature, sea_surface_temperature, total_surface_temperature = (
            _jax_gcm_fields.prepare_surface_temperature_forcing(
                fields["land_surface_temperature"],
                fields["sea_surface_temperature"],
                land_fraction,
            )
        )
        soil_moisture = as_jax_real_array(fields["soil_moisture"], dtype)
        require_active_finite(
            soil_moisture, active_mask=land_fraction > 0, owner="JCM land soil moisture"
        )
        soil_moisture = jnp.where(land_fraction > 0, soil_moisture, 0.0)
        applied_forcing = payload.forcing.copy(
            stl_am=land_surface_temperature.T,
            sea_surface_temperature=blend_jcm_forcing_field(
                payload.forcing.sea_surface_temperature,
                ((1 - prescribed_weight) * sea_surface_temperature).T,
                prescribed_weight.T,
            ),
            soilw_am=soil_moisture.T,
        )
        surface_fields = {
            "land_surface_temperature": land_surface_temperature,
            "sea_surface_temperature": sea_surface_temperature,
            "total_surface_temperature": total_surface_temperature,
        }
    jcm_state, _ = state._step_function(
        payload.jcm_state,
        applied_forcing,
    )
    _require_finite_jax_gcm_state(
        jcm_state,
        component_name=state.name,
        stage="step output",
    )
    mapped_fields = _map_jax_gcm_state_fields(state, jcm_state, constants, dtype)

    # Native JCM removes its transient _forcing_2d diagnostic from the carry.
    # Select the same dates as the last physics step of each saved snapshot;
    # this is pure boundary sampling, without another physics evaluation.
    sampled_surface = _sample_surface_forcing(
        state, payload.jcm_state, applied_forcing, dtype
    )
    land_temperature = sampled_surface["land_surface_temperature"]
    sea_temperature = sampled_surface["sea_surface_temperature"]
    surface_fields.update(
        land_surface_temperature=land_temperature,
        sea_surface_temperature=sea_temperature,
        total_surface_temperature=land_fraction * land_temperature
        + ocean_fraction * sea_temperature,
    )
    if state.seaice_coupling:
        surface_fields.update(
            marine_surface_albedo=as_jax_real_array(
                jcm_state.physics["_mod_radcon"].alb_s, dtype
            ).T,
            marine_ice_fraction=sampled_surface["marine_ice_fraction"],
        )

    return StepResult(
        fields={
            **surface_fields,
            **mapped_fields,
        },
        payload=JCMAtmosphereRuntimePayload(
            jcm_state=jcm_state,
            forcing=payload.forcing,
        ),
    )


def _sample_surface_forcing(
    state: "JCMAtmosphereSetupState",
    initial_state: Any,
    forcing: Any,
    dtype: DTypePolicy,
) -> dict[str, jax.Array]:
    """Average boundary values at the exact physics dates of saved snapshots."""
    return cast(
        dict[str, jax.Array],
        _sample_native_surface_forcing(
            initial_state.time,
            initial_state.step,
            forcing,
            model_seconds=int(state.model_timestep.total_seconds()),
            save_seconds=int(state.save_interval.total_seconds()),
            coupling_seconds=int(state.coupling_timestep.total_seconds()),
            dtype=dtype,
        ),
    )


@partial(
    jax.jit,
    static_argnames=("model_seconds", "save_seconds", "coupling_seconds", "dtype"),
)
def _sample_native_surface_forcing(
    initial_time: Any,
    initial_step: Any,
    forcing: Any,
    *,
    model_seconds: int,
    save_seconds: int,
    coupling_seconds: int,
    dtype: DTypePolicy,
) -> dict[str, jax.Array]:
    """Reuse a fixed-shape native boundary sampler on host and scanned runs."""
    import jax_datetime as jdt
    from jcm.date import DateData

    offsets = jnp.arange(
        save_seconds - model_seconds, coupling_seconds, save_seconds, dtype=jnp.int32
    )

    def sample(offset: jax.Array) -> dict[str, jax.Array]:
        date = DateData.set_date(
            initial_time + jdt.Timedelta(seconds=offset),
            model_step=initial_step + offset // model_seconds,
            dt_seconds=model_seconds,
        )
        selected = forcing.select(date)
        return {
            name: as_jax_real_array(value, dtype).T
            for name, value in (
                ("land_surface_temperature", selected.stl_am),
                ("sea_surface_temperature", selected.sea_surface_temperature),
                ("marine_ice_fraction", selected.sice_am),
            )
        }

    return cast(
        dict[str, jax.Array],
        jax.tree_util.tree_map(
            lambda value: jnp.mean(value, axis=0), jax.vmap(sample)(offsets)
        ),
    )


def step_jax_gcm_component(
    state: "JCMAtmosphereSetupState",
    fields: Mapping[str, Any],
    context: StepContext,
    payload: Any | None,
) -> StepResult:
    """Advance JAXGCM on immutable runtime state."""

    return step_jax_gcm_runtime(
        state,
        fields,
        payload,
        context.constants,
        context.dtype,
    )


__all__ = [
    "JCMAtmosphereRuntimePayload",
    "create_jax_gcm_runtime_payload",
    "jax_gcm_default_field_names",
    "jax_gcm_default_fields",
    "prefill_jax_gcm_runtime_fields",
    "step_jax_gcm_component",
    "step_jax_gcm_runtime",
    "validate_jax_gcm_runtime_state",
]
