"""JAXGCM output cadence and NetCDF writing helpers."""

from __future__ import annotations

import csv
from collections.abc import MutableSequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from vercor.calendar import ModelDateTime
from vercor.host_arrays import runtime_array_to_host
from vercor.jax_logging import LoggerLike, get_default_logger
from vercor.types import RuntimeArray

_TIME_NAME = "time"
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
_TIME_UNITS = "microseconds since 0001-01-01 00:00:00.000000"
_MICROSECONDS_PER_SECOND = 1_000_000
_SECONDS_PER_DAY = 86_400


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


@dataclass(frozen=True)
class _VariableData:
    """Variable payload with source dimensions and host-backed values."""

    dims: tuple[str, ...]
    values: NDArray[Any]


def is_period_end(
    time: datetime | ModelDateTime,
    dt: timedelta,
    frequency: Literal["day", "month", "year"],
) -> bool:
    """Return whether ``time + dt`` crosses the requested calendar boundary."""

    next_time = time + dt

    if frequency == "day":
        return (
            next_time.year != time.year
            or next_time.month != time.month
            or next_time.day != time.day
        )
    if frequency == "month":
        return next_time.year != time.year or next_time.month != time.month

    return next_time.year != time.year


def should_write_period_output(
    *,
    time: datetime | ModelDateTime,
    dt: timedelta,
    output_frequency: str | None,
) -> bool:
    """Return whether JAXGCM should write an output average for this step."""

    if output_frequency is None:
        return True

    if not isinstance(output_frequency, str):
        return False

    frequency = output_frequency.lower()
    if frequency not in ("day", "month", "year"):
        return False

    return is_period_end(
        time=time,
        dt=dt,
        frequency=cast(Literal["day", "month", "year"], frequency),
    )


def _timedelta_to_microseconds(delta: timedelta) -> int:
    return (
        delta.days * _SECONDS_PER_DAY + delta.seconds
    ) * _MICROSECONDS_PER_SECOND + delta.microseconds


def _time_value_and_attrs(
    time: datetime | ModelDateTime,
) -> tuple[NDArray[Any], dict[str, Any]]:
    if isinstance(time, datetime):
        delta = time - datetime(1, 1, 1)
        return (
            np.asarray([_timedelta_to_microseconds(delta)], dtype=np.int64),
            {
                "units": _TIME_UNITS,
                "calendar": "proleptic_gregorian",
                "isoformat": time.isoformat(),
                "day_of_year": time.timetuple().tm_yday,
            },
        )

    origin = type(time)(1, 1, 1, 0, 0, 0, 0, 1)
    model_delta = time - origin
    if not isinstance(model_delta, timedelta):
        raise TypeError("model-calendar output time subtraction must return timedelta")

    calendar = "360_day" if time.fixed_30_day_months else "noleap"
    return (
        np.asarray([_timedelta_to_microseconds(model_delta)], dtype=np.int64),
        {
            "units": _TIME_UNITS,
            "calendar": calendar,
            "isoformat": time.isoformat(),
            "day_of_year": time.day_of_year,
            "days_per_year": time.days_per_year,
            "fixed_30_day_months": int(time.fixed_30_day_months),
        },
    )


def _host_array(value: Any) -> NDArray[Any]:
    return np.asarray(jax.device_get(value))


def _runtime_host_array(value: RuntimeArray) -> NDArray[Any]:
    return runtime_array_to_host(value)


def _vertical_layers(coords: Any) -> int:
    level = _runtime_host_array(coords.vertical.centers)
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


def _coordinate_values(
    *,
    coords: Any,
    output_time: datetime | ModelDateTime,
) -> tuple[dict[str, NDArray[Any]], dict[str, dict[str, Any]]]:
    lon, sin_lat = coords.horizontal.nodal_axes
    lon_k, lat_k = coords.horizontal.modal_axes
    level = _runtime_host_array(coords.vertical.centers)
    layers = _vertical_layers(coords)
    time_values, time_attrs = _time_value_and_attrs(output_time)

    coordinate_values = {
        _TIME_NAME: time_values,
        _LON_NAME: _runtime_host_array(lon) * 180.0 / np.pi,
        _LAT_NAME: np.arcsin(_runtime_host_array(sin_lat)) * 180.0 / np.pi,
        _LON_MODE_NAME: _runtime_host_array(lon_k),
        _LAT_MODE_NAME: _runtime_host_array(lat_k),
        _LEVEL_NAME: level,
        _WVI_NAME: np.asarray([1, 2], dtype=np.int64),
        _HSG_LEVEL_NAME: np.arange(layers + 1, dtype=np.int64),
    }
    if layers != 1:
        coordinate_values[_SURFACE_NAME] = np.ones(1, dtype=float)

    coordinate_attrs = {
        _TIME_NAME: time_attrs,
        _LON_NAME: {"units": "degrees_east"},
        _LAT_NAME: {"units": "degrees_north"},
    }
    return coordinate_values, coordinate_attrs


def _additional_coordinate_values(coords: Any) -> dict[str, NDArray[Any]]:
    layers = _vertical_layers(coords)
    values: dict[str, NDArray[Any]] = {
        _WVI_NAME: np.asarray([1, 2], dtype=np.int64),
        _HSG_LEVEL_NAME: np.arange(layers + 1, dtype=np.int64),
    }
    if layers != 1:
        values[_SURFACE_NAME] = np.ones(1, dtype=float)
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
        tuple(_host_array(sin_lat).shape): (_LAT_NAME,),
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
) -> dict[str, _VariableData]:
    dynamics_predictions = _float0s_to_nans(prediction.dynamics)
    physics_predictions = _float0s_to_nans(prediction.physics)
    time_values = _host_array(prediction.times)
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

    variables: dict[str, _VariableData] = {}
    for name, value in _iter_data_items(data):
        values = _host_array(value)
        dims = shape_to_dims.get(tuple(values.shape))
        if dims is None:
            raise ValueError(
                f"Value of shape {values.shape} for variable {name!r} is not recognized."
            )
        if _TIME_NAME not in dims:
            raise ValueError(f"Variable {name!r} does not include a time dimension.")
        variables[name] = _VariableData(dims=dims, values=values)
    return variables


def _concatenate_prediction_variables(
    prediction_variables: list[dict[str, _VariableData]],
) -> dict[str, _VariableData]:
    if not prediction_variables:
        raise ValueError("JAXGCM average output requires at least one prediction.")

    output: dict[str, _VariableData] = {}
    variable_names = tuple(prediction_variables[0].keys())
    for variables in prediction_variables[1:]:
        if tuple(variables.keys()) != variable_names:
            raise ValueError("JAXGCM prediction variables changed across outputs.")

    for name in variable_names:
        first = prediction_variables[0][name]
        time_axis = first.dims.index(_TIME_NAME)
        arrays = [variables[name].values for variables in prediction_variables]
        for variables in prediction_variables[1:]:
            if variables[name].dims != first.dims:
                raise ValueError(f"JAXGCM variable {name!r} dimensions changed.")
        output[name] = _VariableData(
            dims=first.dims,
            values=np.concatenate(arrays, axis=time_axis),
        )
    return output


def _mean_and_transpose_variable(variable: _VariableData) -> _VariableData:
    time_axis = variable.dims.index(_TIME_NAME)
    values = np.nanmean(variable.values, axis=time_axis, keepdims=True)
    ordered_dims = tuple(
        dim for dim in _CANONICAL_DIM_ORDER if dim in variable.dims
    ) + tuple(dim for dim in variable.dims if dim not in _CANONICAL_DIM_ORDER)
    axes = tuple(variable.dims.index(dim) for dim in ordered_dims)
    if axes != tuple(range(len(axes))):
        values = np.transpose(values, axes=axes)
    return _VariableData(dims=ordered_dims, values=values)


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
    variables: dict[str, _VariableData],
    unit_metadata: dict[str, dict[str, str]],
) -> None:
    coordinate_values, coordinate_attrs = _coordinate_values(
        coords=coords,
        output_time=output_time,
    )
    with h5netcdf.File(output, "w") as outfile:
        outfile.attrs["Conventions"] = "CF-1.8"
        for name, values in coordinate_values.items():
            outfile.dimensions[name] = values.shape[0]
            coordinate = outfile.create_variable(name, (name,), data=values)
            for attr_name, attr_value in coordinate_attrs.get(name, {}).items():
                if attr_value is not None:
                    coordinate.attrs[attr_name] = attr_value

        for name, variable_data in variables.items():
            for dim, size in zip(variable_data.dims, variable_data.values.shape):
                if dim not in outfile.dimensions:
                    outfile.dimensions[dim] = size
            variable = outfile.create_variable(
                name,
                variable_data.dims,
                data=variable_data.values,
            )
            for attr_name, attr_value in unit_metadata.get(name, {}).items():
                if attr_value:
                    variable.attrs[attr_name] = attr_value


def write_jax_gcm_averages_output(
    predictions: MutableSequence[Any],
    output: str,
    *,
    coords: Any,
    output_time: datetime | ModelDateTime,
    physics_module: _PhysicsModuleLike | None = None,
    logger: LoggerLike | None = None,
) -> None:
    """Write mean JAXGCM predictions to NetCDF and clear the prediction buffer."""

    log = logger if logger is not None else get_default_logger()
    log.info(f"Output file: {output:s}")

    selected_physics_module = physics_module or _default_physics_module()
    prediction_variables = [
        _prediction_to_variables(
            cast(_PredictionLike, prediction),
            coords=coords,
            physics_module=selected_physics_module,
        )
        for prediction in predictions
    ]
    mean_variables = {
        name: _mean_and_transpose_variable(variable)
        for name, variable in _concatenate_prediction_variables(
            prediction_variables
        ).items()
    }
    _write_netcdf(
        output=output,
        coords=coords,
        output_time=output_time,
        variables=mean_variables,
        unit_metadata=_unit_metadata(selected_physics_module),
    )

    predictions.clear()
