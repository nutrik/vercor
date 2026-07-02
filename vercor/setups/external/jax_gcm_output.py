"""JAXGCM output extraction and metadata helpers."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.dtypes import as_jax_index_array, as_jax_real_array, jax_index_dtype
from vercor.jax_logging import LoggerLike
from vercor.output.adapters import ComponentOutputAdapter
from vercor.output.datasets import time_coordinate_variable
from vercor.output.time import TIME_NAME
from vercor.output.variables import OutputVariable
from vercor.types import RuntimeArray

JAX_GCM_TIME_DIM = TIME_NAME
_LEVEL_NAME = "level"
_LON_NAME = "lon"
_LAT_NAME = "lat"
_LON_MODE_NAME = "longitudinal_mode"
_LAT_MODE_NAME = "total_wavenumber"
_WVI_NAME = "wvi_id"
_HSG_LEVEL_NAME = "hsg_level"
_SURFACE_NAME = "surface"
JAX_GCM_AVERAGE_EMPTY_ERROR_MESSAGE = (
    "JAXGCM average output requires at least one prediction."
)
JAX_GCM_OUTPUT_DIMENSION_ORDER = (
    JAX_GCM_TIME_DIM,
    _WVI_NAME,
    _HSG_LEVEL_NAME,
    _SURFACE_NAME,
    _LEVEL_NAME,
    _LAT_NAME,
    _LON_NAME,
    _LON_MODE_NAME,
    _LAT_MODE_NAME,
)


class _PredictionLike(Protocol):
    """JCM prediction fields consumed by the direct NetCDF writer."""

    dynamics: Any
    physics: Any
    times: RuntimeArray


class _PhysicsModuleLike(Protocol):
    """Subset of the JCM physics module output API used by this writer."""

    UNITS_TABLE_CSV_PATH: str | Path | None

    def cache_coords(self, coords: Any) -> None: ...

    def data_struct_to_dict(
        self,
        struct: Any,
        nodal_shape: tuple[int, ...],
    ) -> dict[str, Any]: ...


def _vertical_layers(coords: Any) -> int:
    level = as_jax_real_array(coords.vertical.centers)
    layers = getattr(coords.vertical, "layers", None)
    if layers is None:
        return int(level.shape[0])
    return int(cast(Any, layers))


def _float0_leaf_to_nan(value: Any) -> Any:
    if getattr(value, "dtype", None) == jax.dtypes.float0:
        return jnp.full_like(value, jnp.nan, dtype=float)
    return value


def _float0s_to_nans(pytree: Any) -> Any:
    return jax.tree_util.tree_map(_float0_leaf_to_nan, pytree)


def jax_gcm_coordinate_variables(
    *,
    coords: Any,
    output_time: datetime | ModelDateTime,
) -> dict[str, OutputVariable]:
    """Return NetCDF coordinate variables for JAXGCM period output."""

    lon, sin_lat = coords.horizontal.nodal_axes
    lon_k, lat_k = coords.horizontal.modal_axes
    level = as_jax_real_array(coords.vertical.centers)
    layers = _vertical_layers(coords)

    coordinate_variables = {
        JAX_GCM_TIME_DIM: time_coordinate_variable(
            output_time,
            time_dim=JAX_GCM_TIME_DIM,
        ),
        _LON_NAME: OutputVariable(
            (_LON_NAME,),
            as_jax_real_array(lon) * 180.0 / jnp.pi,
            {"units": "degrees_east"},
        ),
        _LAT_NAME: OutputVariable(
            (_LAT_NAME,),
            jnp.arcsin(as_jax_real_array(sin_lat)) * 180.0 / jnp.pi,
            {"units": "degrees_north"},
        ),
        _LON_MODE_NAME: OutputVariable(
            (_LON_MODE_NAME,),
            as_jax_index_array(lon_k),
        ),
        _LAT_MODE_NAME: OutputVariable(
            (_LAT_MODE_NAME,),
            as_jax_index_array(lat_k),
        ),
        _LEVEL_NAME: OutputVariable((_LEVEL_NAME,), level),
        _WVI_NAME: OutputVariable((_WVI_NAME,), as_jax_index_array([1, 2])),
        _HSG_LEVEL_NAME: OutputVariable(
            (_HSG_LEVEL_NAME,),
            jnp.arange(layers + 1, dtype=jax_index_dtype()),
        ),
    }
    if layers != 1:
        coordinate_variables[_SURFACE_NAME] = OutputVariable(
            (_SURFACE_NAME,),
            as_jax_real_array([1.0]),
        )
    return coordinate_variables


def _additional_coordinate_values(coords: Any) -> dict[str, jax.Array]:
    layers = _vertical_layers(coords)
    values: dict[str, jax.Array] = {
        _WVI_NAME: as_jax_index_array([1, 2]),
        _HSG_LEVEL_NAME: jnp.arange(layers + 1, dtype=jax_index_dtype()),
    }
    if layers != 1:
        values[_SURFACE_NAME] = as_jax_real_array([1.0])
    return values


def _infer_shape_to_dims(
    *,
    coords: Any,
    time_shape: tuple[int, ...],
) -> dict[tuple[int, ...], tuple[str, ...]]:
    lon, sin_lat = coords.horizontal.nodal_axes
    _ = lon
    layers = _vertical_layers(coords)
    modal_shape = tuple(coords.horizontal.modal_shape)
    nodal_shape = tuple(coords.horizontal.nodal_shape)

    basic_shape_to_dims: dict[tuple[int, ...], tuple[str, ...]] = {
        tuple(): tuple(),
        (layers,) + modal_shape: (_LEVEL_NAME, _LON_MODE_NAME, _LAT_MODE_NAME),
        (layers,) + nodal_shape: (_LEVEL_NAME, _LON_NAME, _LAT_NAME),
        nodal_shape: (_LON_NAME, _LAT_NAME),
        modal_shape: (_LON_MODE_NAME, _LAT_MODE_NAME),
        (layers,): (_LEVEL_NAME,),
        tuple(as_jax_real_array(sin_lat).shape): (_LAT_NAME,),
        tuple(coords.surface_nodal_shape): (_LON_NAME, _LAT_NAME),
    }

    for dim, value in _additional_coordinate_values(coords).items():
        value_shape = tuple(value.shape)
        if value.ndim != 1:
            raise ValueError(
                "`additional_coords` must be 1d vectors, but got "
                f"shape={value.shape} for dim={dim}"
            )
        if value_shape == (layers,):
            raise ValueError(
                f"additional coordinate dim={dim} collides with level shape"
            )
        basic_shape_to_dims[value_shape + modal_shape] = (
            dim,
            _LON_MODE_NAME,
            _LAT_MODE_NAME,
        )
        basic_shape_to_dims[value_shape + nodal_shape] = (
            dim,
            _LON_NAME,
            _LAT_NAME,
        )
        basic_shape_to_dims[value_shape] = (dim,)
        basic_shape_to_dims[(layers,) + value_shape] = (_LEVEL_NAME, dim)

    return {
        time_shape + shape: (JAX_GCM_TIME_DIM,) + dims
        for shape, dims in basic_shape_to_dims.items()
    }


def _mapping_from_struct(struct: Any) -> dict[str, Any]:
    if hasattr(struct, "asdict"):
        return dict(struct.asdict())
    if isinstance(struct, dict):
        return dict(struct)
    return {
        key: value for key, value in vars(struct).items() if not key.startswith("_")
    }


def _iter_data_items(data: dict[str, Any]) -> list[tuple[str, Any]]:
    prognostic_keys = [
        key for key in data.keys() if key not in {"tracers", "diagnostics"}
    ]
    tracer_data = data.get("tracers", {})
    diagnostic_data = data.get("diagnostics", {})
    tracer_keys = set(tracer_data.keys()) if isinstance(tracer_data, dict) else set()
    diagnostic_keys = (
        set(diagnostic_data.keys()) if isinstance(diagnostic_data, dict) else set()
    )

    if not set(prognostic_keys).isdisjoint(tracer_keys):
        raise ValueError(
            "Tracer names collide with prognostic variables",
            f"Tracers: {tracer_keys}; prognostics: {set(prognostic_keys)}",
        )
    if not set(prognostic_keys).isdisjoint(diagnostic_keys):
        raise ValueError(
            "Diagnostic names collide with prognostic variables",
            f"Diagnostic: {diagnostic_keys}; prognostics: {set(prognostic_keys)}",
        )

    items = [(key, data[key]) for key in prognostic_keys]
    if isinstance(tracer_data, dict):
        items.extend(tracer_data.items())
    if isinstance(diagnostic_data, dict):
        items.extend(diagnostic_data.items())
    return items


def _default_physics_module() -> _PhysicsModuleLike:
    from jcm.physics.speedy.speedy_physics import SpeedyPhysics

    return cast(_PhysicsModuleLike, SpeedyPhysics())


def jax_gcm_prediction_output_variables(
    prediction: Any,
    *,
    coords: Any,
    physics_module: _PhysicsModuleLike | None = None,
) -> dict[str, OutputVariable]:
    """Return JAXGCM prediction variables ready for period accumulation."""

    selected_physics_module = physics_module or _default_physics_module()
    prediction_values = cast(_PredictionLike, prediction)
    dynamics_predictions = _float0s_to_nans(prediction_values.dynamics)
    physics_predictions = _float0s_to_nans(prediction_values.physics)
    time_values = as_jax_real_array(prediction_values.times)
    time_shape = tuple(time_values.shape)
    shape_to_dims = _infer_shape_to_dims(coords=coords, time_shape=time_shape)

    nodal_shape = (
        _vertical_layers(coords),
        *tuple(coords.horizontal.nodal_shape),
    )
    selected_physics_module.cache_coords(coords)
    data = _mapping_from_struct(dynamics_predictions)
    data.update(
        selected_physics_module.data_struct_to_dict(
            physics_predictions,
            nodal_shape=nodal_shape,
        )
    )

    variables: dict[str, OutputVariable] = {}
    for name, value in _iter_data_items(data):
        values = as_jax_real_array(value)
        dims = shape_to_dims.get(tuple(values.shape))
        if dims is None:
            raise ValueError(
                f"Value of shape {values.shape} for variable {name!r} "
                "is not recognized."
            )
        if JAX_GCM_TIME_DIM not in dims:
            raise ValueError(f"Variable {name!r} does not include a time dimension.")
        variables[name] = OutputVariable(dims=dims, values=values)
    return variables


def jax_gcm_state_snapshot_output_variables(
    jcm_state: Any,
    *,
    coords: Any,
    physics_module: _PhysicsModuleLike | None = None,
) -> dict[str, OutputVariable]:
    """Return final JAXGCM state variables ready for snapshot output."""

    prediction = SimpleNamespace(
        dynamics=jax.tree_util.tree_map(_with_leading_time_dim, jcm_state.prog),
        physics=jax.tree_util.tree_map(_with_leading_time_dim, jcm_state.phydata),
        times=as_jax_real_array([0.0]),
    )
    return jax_gcm_prediction_output_variables(
        prediction,
        coords=coords,
        physics_module=physics_module,
    )


def _with_leading_time_dim(value: Any) -> Any:
    return jnp.asarray(value)[jnp.newaxis, ...]


def _read_units_table(path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            variable = row.get("Variable")
            if variable is None:
                continue
            metadata[variable] = {
                "units": row.get("Units", ""),
                "description": row.get("Description", ""),
            }
    return metadata


def jax_gcm_unit_metadata(
    physics_module: _PhysicsModuleLike | None = None,
) -> dict[str, dict[str, str]]:
    """Return JAXGCM unit metadata from bundled dynamics and physics tables."""

    selected_physics_module = physics_module or _default_physics_module()
    dynamics_resource = resources.files("jcm").joinpath("dynamics_units_table.csv")
    with resources.as_file(dynamics_resource) as dynamics_path:
        metadata = _read_units_table(dynamics_path)

    physics_units_path = selected_physics_module.UNITS_TABLE_CSV_PATH
    if physics_units_path is not None:
        metadata.update(_read_units_table(Path(physics_units_path)))
    return metadata


def jax_gcm_data_variables_with_unit_metadata(
    variables: Mapping[str, OutputVariable],
    unit_metadata: Mapping[str, Mapping[str, str]],
) -> dict[str, OutputVariable]:
    """Return JAXGCM data variables with unit metadata attached."""

    return {
        name: OutputVariable(
            dims=variable_data.dims,
            values=variable_data.values,
            attrs={
                **dict(variable_data.attrs),
                **{
                    attr_name: attr_value
                    for attr_name, attr_value in unit_metadata.get(name, {}).items()
                    if attr_value
                },
            },
        )
        for name, variable_data in variables.items()
    }


def record_jax_gcm_period_output(
    adapter: ComponentOutputAdapter,
    prediction: Any,
    *,
    coords: Any,
    physics_module: _PhysicsModuleLike | None = None,
    output_time: datetime | ModelDateTime,
    dt: timedelta,
    output_frequency: str | None,
    logger: LoggerLike | None = None,
) -> bool:
    """Record one JAXGCM prediction and write a period average when due."""

    variables = jax_gcm_prediction_output_variables(
        prediction,
        coords=coords,
        physics_module=physics_module,
    )

    def build_coordinate_variables(
        mean_variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        _ = mean_variables
        return jax_gcm_coordinate_variables(
            coords=coords,
            output_time=output_time,
        )

    def build_data_variables(
        mean_variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        unit_metadata = jax_gcm_unit_metadata(physics_module)
        return jax_gcm_data_variables_with_unit_metadata(
            mean_variables,
            unit_metadata,
        )

    return adapter.record_period_average_if_due(
        variables,
        summation_dim=JAX_GCM_TIME_DIM,
        time=output_time,
        dt=dt,
        output_frequency=output_frequency,
        output=lambda time: f"jcm.averages.{time.strftime('%Y-%m-%d')}.nc",
        build_coordinate_variables=build_coordinate_variables,
        build_data_variables=build_data_variables,
        logger=logger,
    )


def write_jax_gcm_snapshot_output(
    state: Any,
    component_state: Any,
    output: Path,
    output_time: datetime | ModelDateTime,
    logger: LoggerLike | None = None,
) -> None:
    """Write one final JAXGCM state snapshot through the shared output adapter."""

    payload = getattr(component_state, "runtime_payload", None)
    jcm_state = getattr(payload, "jcm_state", None)
    if jcm_state is None:
        jcm_state = getattr(state, "_state", None)
    if jcm_state is None:
        raise ValueError("JAXGCM snapshot output requires a final JCM state.")

    physics_module = getattr(state.model, "physics", None)
    variables = jax_gcm_state_snapshot_output_variables(
        jcm_state,
        coords=state.model.coords,
        physics_module=physics_module,
    )
    state.output_adapter.record_snapshot(
        variables,
        summation_dim=JAX_GCM_TIME_DIM,
        time=output_time,
    )
    unit_metadata = jax_gcm_unit_metadata(physics_module)

    def build_coordinate_variables(
        snapshot_variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        _ = snapshot_variables
        return jax_gcm_coordinate_variables(
            coords=state.model.coords,
            output_time=output_time,
        )

    def build_data_variables(
        snapshot_variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        return jax_gcm_data_variables_with_unit_metadata(
            snapshot_variables,
            unit_metadata,
        )

    state.output_adapter.write_snapshot(
        str(output),
        build_coordinate_variables=build_coordinate_variables,
        build_data_variables=build_data_variables,
        logger=logger,
    )


__all__ = [
    "JAX_GCM_AVERAGE_EMPTY_ERROR_MESSAGE",
    "JAX_GCM_OUTPUT_DIMENSION_ORDER",
    "JAX_GCM_TIME_DIM",
    "jax_gcm_coordinate_variables",
    "jax_gcm_data_variables_with_unit_metadata",
    "jax_gcm_prediction_output_variables",
    "jax_gcm_state_snapshot_output_variables",
    "jax_gcm_unit_metadata",
    "record_jax_gcm_period_output",
    "write_jax_gcm_snapshot_output",
]
