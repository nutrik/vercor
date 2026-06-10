"""JAXGCM output extraction and period-average writing helpers."""

from __future__ import annotations

import csv
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.dtypes import as_jax_index_array, as_jax_real_array, jax_index_dtype
from vercor.jax_logging import LoggerLike, get_default_logger
from vercor.output.netcdf import write_netcdf_dataset
from vercor.output.period_averages import (
    PeriodAverageAccumulator,
    PeriodAverageSample,
)
from vercor.output.time import TIME_NAME, output_time_value_and_attrs
from vercor.output.variables import OutputVariable
from vercor.types import RuntimeArray

_TIME_NAME = TIME_NAME
_LEVEL_NAME = "level"
_LON_NAME = "lon"
_LAT_NAME = "lat"
_LON_MODE_NAME = "longitudinal_mode"
_LAT_MODE_NAME = "total_wavenumber"
_WVI_NAME = "wvi_id"
_HSG_LEVEL_NAME = "hsg_level"
_SURFACE_NAME = "surface"
_CANONICAL_DIM_ORDER = (
    _TIME_NAME,
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


def _coordinate_variables(
    *,
    coords: Any,
    output_time: datetime | ModelDateTime,
) -> dict[str, OutputVariable]:
    lon, sin_lat = coords.horizontal.nodal_axes
    lon_k, lat_k = coords.horizontal.modal_axes
    level = as_jax_real_array(coords.vertical.centers)
    layers = _vertical_layers(coords)
    time_values, time_attrs = output_time_value_and_attrs(output_time)

    coordinate_variables = {
        _TIME_NAME: OutputVariable((_TIME_NAME,), time_values, time_attrs),
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
        time_shape + shape: (_TIME_NAME,) + dims
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


def _prediction_to_variables(
    prediction: _PredictionLike,
    *,
    coords: Any,
    physics_module: _PhysicsModuleLike,
) -> dict[str, OutputVariable]:
    dynamics_predictions = _float0s_to_nans(prediction.dynamics)
    physics_predictions = _float0s_to_nans(prediction.physics)
    time_values = as_jax_real_array(prediction.times)
    time_shape = tuple(time_values.shape)
    shape_to_dims = _infer_shape_to_dims(coords=coords, time_shape=time_shape)

    nodal_shape = (
        _vertical_layers(coords),
        *tuple(coords.horizontal.nodal_shape),
    )
    physics_module.cache_coords(coords)
    data = _mapping_from_struct(dynamics_predictions)
    data.update(
        physics_module.data_struct_to_dict(
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
        if _TIME_NAME not in dims:
            raise ValueError(f"Variable {name!r} does not include a time dimension.")
        variables[name] = OutputVariable(dims=dims, values=values)
    return variables


def accumulate_jax_gcm_period_prediction(
    accumulator: PeriodAverageAccumulator,
    prediction: Any,
    *,
    coords: Any,
    physics_module: _PhysicsModuleLike | None = None,
) -> None:
    """Accumulate one JAXGCM prediction block into period running sums."""

    selected_physics_module = physics_module or _default_physics_module()
    variables = _prediction_to_variables(
        cast(_PredictionLike, prediction),
        coords=coords,
        physics_module=selected_physics_module,
    )
    accumulator.add_samples(
        {
            name: PeriodAverageSample(
                dims=variable.dims,
                values=variable.values,
                attrs=variable.attrs,
            )
            for name, variable in variables.items()
        },
        summation_dim=_TIME_NAME,
    )


def _period_mean_sample_to_variable(sample: PeriodAverageSample) -> OutputVariable:
    dims = (_TIME_NAME, *sample.dims)
    values = as_jax_real_array(sample.values)[jnp.newaxis, ...]
    ordered_dims = tuple(dim for dim in _CANONICAL_DIM_ORDER if dim in dims) + tuple(
        dim for dim in dims if dim not in _CANONICAL_DIM_ORDER
    )
    axes = tuple(dims.index(dim) for dim in ordered_dims)
    if axes != tuple(range(len(axes))):
        values = jnp.transpose(values, axes=axes)
    return OutputVariable(dims=ordered_dims, values=values, attrs=dict(sample.attrs))


def _jax_gcm_mean_samples(
    accumulator: PeriodAverageAccumulator,
) -> dict[str, PeriodAverageSample]:
    try:
        return accumulator.mean_samples()
    except ValueError as exc:
        raise ValueError(
            "JAXGCM average output requires at least one prediction."
        ) from exc


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


def _unit_metadata(physics_module: _PhysicsModuleLike) -> dict[str, dict[str, str]]:
    dynamics_resource = resources.files("jcm").joinpath("dynamics_units_table.csv")
    with resources.as_file(dynamics_resource) as dynamics_path:
        metadata = _read_units_table(dynamics_path)

    physics_units_path = physics_module.UNITS_TABLE_CSV_PATH
    if physics_units_path is not None:
        metadata.update(_read_units_table(Path(physics_units_path)))
    return metadata


def _write_netcdf(
    *,
    output: str,
    coords: Any,
    output_time: datetime | ModelDateTime,
    variables: dict[str, OutputVariable],
    unit_metadata: dict[str, dict[str, str]],
) -> None:
    write_netcdf_dataset(
        output=output,
        coordinate_variables=_coordinate_variables(
            coords=coords,
            output_time=output_time,
        ),
        data_variables={
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
        },
    )


def write_jax_gcm_averages_output(
    accumulator: PeriodAverageAccumulator,
    output: str,
    *,
    coords: Any,
    output_time: datetime | ModelDateTime,
    physics_module: _PhysicsModuleLike | None = None,
    logger: LoggerLike | None = None,
) -> None:
    """Write mean JAXGCM predictions to NetCDF and clear the accumulator."""

    log = logger if logger is not None else get_default_logger()
    log.info(f"Output file: {output:s}")

    selected_physics_module = physics_module or _default_physics_module()
    mean_variables = {
        name: _period_mean_sample_to_variable(sample)
        for name, sample in _jax_gcm_mean_samples(accumulator).items()
    }
    _write_netcdf(
        output=output,
        coords=coords,
        output_time=output_time,
        variables=mean_variables,
        unit_metadata=_unit_metadata(selected_physics_module),
    )

    accumulator.clear()


__all__ = [
    "accumulate_jax_gcm_period_prediction",
    "write_jax_gcm_averages_output",
]
