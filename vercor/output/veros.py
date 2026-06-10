"""Veros period-output extraction and average-file helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.dtypes import as_jax_index_array, as_jax_real_array
from vercor.host_arrays import array_to_host
from vercor.jax_logging import LoggerLike, get_default_logger
from vercor.output.netcdf import write_netcdf_dataset
from vercor.output.period_averages import (
    PeriodAverageAccumulator,
    PeriodAverageSample,
)
from vercor.output.time import TIME_NAME, output_time_value_and_attrs
from vercor.output.variables import OutputVariable
from vercor.setups.external.veros_runtime_settings import configure_veros_runtime

configure_veros_runtime()

from veros import variables as veros_variables  # noqa: E402

_TIME_NAME = TIME_NAME
_TIMESTEP_DIM = "timesteps"
_GHOST_DIMS = ("xt", "yt", "xu", "yu")

VerosOutputVariable = OutputVariable


def normalize_veros_output_variables(
    output_variables: Sequence[str] | None,
    *,
    settings: Any,
) -> tuple[str, ...]:
    """Return validated Veros output variable names in user-provided order."""

    if output_variables is None:
        return ()
    if isinstance(output_variables, str):
        raise ValueError("Veros output_variables must be a sequence of names.")

    normalized = tuple(output_variables)
    for name in normalized:
        if not isinstance(name, str):
            raise ValueError("Veros output_variables entries must be strings.")
        variable = veros_variables.VARIABLES.get(name)
        if variable is None:
            raise ValueError(f"Unknown Veros output variable {name!r}.")
        if not bool(_resolve_metadata(variable.active, settings)):
            raise ValueError(
                f"Veros output variable {name!r} is inactive for current settings."
            )
    return normalized


def extract_veros_output_snapshot(
    veros_state: Any,
    output_variables: Sequence[str],
) -> dict[str, OutputVariable]:
    """Extract selected Veros variables at the current state timestep."""

    selected_variables = normalize_veros_output_variables(
        output_variables,
        settings=veros_state.settings,
    )
    return {name: _extract_variable(veros_state, name) for name in selected_variables}


def accumulate_veros_output_snapshot(
    accumulator: PeriodAverageAccumulator,
    snapshot: Mapping[str, OutputVariable],
) -> None:
    """Accumulate one selected Veros output snapshot for period averaging."""

    accumulator.add_samples(
        {
            name: PeriodAverageSample(
                dims=variable.dims,
                values=variable.values,
                attrs=variable.attrs,
            )
            for name, variable in snapshot.items()
        }
    )


def accumulate_veros_period_state(
    accumulator: PeriodAverageAccumulator,
    veros_state: Any,
    output_variables: Sequence[str],
) -> None:
    """Extract and accumulate selected Veros state variables for a period."""

    if not output_variables:
        return

    accumulate_veros_output_snapshot(
        accumulator,
        extract_veros_output_snapshot(veros_state, output_variables),
    )


def write_veros_averages_output(
    accumulator: PeriodAverageAccumulator,
    output: str,
    *,
    veros_state: Any,
    output_time: datetime | ModelDateTime,
    logger: LoggerLike | None = None,
) -> None:
    """Write mean Veros output snapshots to NetCDF and clear the accumulator."""

    log = logger if logger is not None else get_default_logger()
    log.info(f"Output file: {output:s}")

    mean_variables = _mean_accumulated_variables(accumulator)
    _write_netcdf(
        output=output,
        veros_state=veros_state,
        output_time=output_time,
        variables=mean_variables,
    )
    accumulator.clear()


def _resolve_metadata(value: Any, settings: Any) -> Any:
    if callable(value):
        return value(settings)
    return value


def _variable_definition(name: str) -> Any:
    variable = veros_variables.VARIABLES.get(name)
    if variable is None:
        raise ValueError(f"Unknown Veros output variable {name!r}.")
    return variable


def _resolved_dims(variable: Any, settings: Any, name: str) -> tuple[str, ...]:
    dims = _resolve_metadata(variable.dims, settings)
    if dims is None:
        return ()
    if not isinstance(dims, tuple) or not all(isinstance(dim, str) for dim in dims):
        raise ValueError(f"Veros output variable {name!r} has invalid dimensions.")
    return dims


def _attrs_for_variable(variable: Any, settings: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    units = _resolve_metadata(variable.units, settings)
    if units:
        attrs["units"] = units
    long_description = _resolve_metadata(variable.long_description, settings)
    if long_description:
        attrs["long_name"] = long_description
    for name, value in variable.extra_attributes.items():
        if value is not None:
            attrs[name] = int(value) if isinstance(value, bool) else value
    return attrs


def _current_timestep_index(vs: Any) -> int:
    return int(array_to_host(as_jax_index_array(vs.tau)))


def _drop_timestep_dim(
    values: jnp.ndarray,
    dims: tuple[str, ...],
    vs: Any,
) -> tuple[jnp.ndarray, tuple[str, ...]]:
    if _TIMESTEP_DIM not in dims:
        return values, dims

    time_axis = dims.index(_TIMESTEP_DIM)
    current_values = jnp.take(values, _current_timestep_index(vs), axis=time_axis)
    return current_values, dims[:time_axis] + dims[time_axis + 1 :]  # noqa: E203


def _remove_ghost_cells(
    values: jnp.ndarray,
    dims: tuple[str, ...],
) -> jnp.ndarray:
    if not dims:
        return values
    slices = tuple(slice(2, -2) if dim in _GHOST_DIMS else slice(None) for dim in dims)
    return values[slices]


def _extract_variable(veros_state: Any, name: str) -> OutputVariable:
    vs = veros_state.variables
    variable = _variable_definition(name)
    dims = _resolved_dims(variable, veros_state.settings, name)
    values = as_jax_real_array(getattr(vs, name))
    values, dims = _drop_timestep_dim(values, dims, vs)
    values = _remove_ghost_cells(values, dims)
    if values.ndim != len(dims):
        raise ValueError(
            f"Veros output variable {name!r} has shape {values.shape} "
            f"but dimensions {dims}."
        )
    return OutputVariable(
        dims=dims,
        values=values,
        attrs=_attrs_for_variable(variable, veros_state.settings),
    )


def _extract_coordinate_variable(veros_state: Any, dim: str) -> OutputVariable:
    variable = _variable_definition(dim)
    dims = _resolved_dims(variable, veros_state.settings, dim)
    if dims != (dim,):
        raise ValueError(f"Veros coordinate {dim!r} must have dimensions ({dim!r},).")
    if not hasattr(veros_state.variables, dim):
        raise ValueError(f"Veros coordinate variable {dim!r} is missing.")
    values = _remove_ghost_cells(
        as_jax_real_array(getattr(veros_state.variables, dim)), dims
    )
    return OutputVariable(
        dims=dims,
        values=values,
        attrs=_attrs_for_variable(variable, veros_state.settings),
    )


def _mean_accumulated_variables(
    accumulator: PeriodAverageAccumulator,
) -> dict[str, OutputVariable]:
    try:
        mean_samples = accumulator.mean_samples()
    except ValueError as exc:
        raise ValueError(
            "Veros average output requires at least one prediction."
        ) from exc

    return {
        name: _mean_sample_to_output_variable(sample)
        for name, sample in mean_samples.items()
    }


def _mean_sample_to_output_variable(sample: PeriodAverageSample) -> OutputVariable:
    axes = tuple(reversed(range(sample.values.ndim)))
    values = as_jax_real_array(sample.values)
    if axes != tuple(range(sample.values.ndim)):
        values = jnp.transpose(values, axes=axes)

    return OutputVariable(
        dims=(_TIME_NAME, *reversed(sample.dims)),
        values=values[jnp.newaxis, ...],
        attrs=dict(sample.attrs),
    )


def _used_coordinate_dims(
    variables: Mapping[str, OutputVariable],
) -> tuple[str, ...]:
    used: list[str] = []
    for variable in variables.values():
        for dim in variable.dims:
            if dim != _TIME_NAME and dim not in used:
                used.append(dim)
    return tuple(used)


def _coordinate_variables(
    *,
    veros_state: Any,
    output_time: datetime | ModelDateTime,
    variables: Mapping[str, OutputVariable],
) -> dict[str, OutputVariable]:
    time_values, time_attrs = output_time_value_and_attrs(output_time)
    coordinate_variables = {
        _TIME_NAME: OutputVariable((_TIME_NAME,), time_values, time_attrs)
    }
    for dim in _used_coordinate_dims(variables):
        coordinate_variables[dim] = _extract_coordinate_variable(veros_state, dim)
    return coordinate_variables


def _write_netcdf(
    *,
    output: str,
    veros_state: Any,
    output_time: datetime | ModelDateTime,
    variables: Mapping[str, OutputVariable],
) -> None:
    write_netcdf_dataset(
        output=output,
        coordinate_variables=_coordinate_variables(
            veros_state=veros_state,
            output_time=output_time,
            variables=variables,
        ),
        data_variables=variables,
    )


__all__ = [
    "VerosOutputVariable",
    "accumulate_veros_output_snapshot",
    "accumulate_veros_period_state",
    "extract_veros_output_snapshot",
    "normalize_veros_output_variables",
    "write_veros_averages_output",
]
