"""Veros period-output extraction and coordinate helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.dtypes import as_jax_index_array, as_jax_real_array
from vercor.host_arrays import array_to_host
from vercor.jax_logging import LoggerLike
from vercor.output.adapters import ComponentOutputAdapter
from vercor.output.datasets import time_coordinate_variable, used_dimension_names
from vercor.output.time import TIME_NAME
from vercor.output.variables import OutputVariable
from vercor.setups.external.veros_runtime_settings import configure_veros_runtime

configure_veros_runtime()

from veros import variables as veros_variables  # noqa: E402

VEROS_TIME_DIM = TIME_NAME
VEROS_AVERAGE_EMPTY_ERROR_MESSAGE = (
    "Veros average output requires at least one prediction."
)
_TIMESTEP_DIM = "timesteps"
_GHOST_DIMS = ("xt", "yt", "xu", "yu")


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


def veros_average_value_dims(sample: OutputVariable) -> tuple[str, ...]:
    """Return Veros NetCDF value dimension order for one mean sample."""

    return tuple(reversed(sample.dims))


def veros_average_coordinate_variables(
    *,
    veros_state: Any,
    output_time: datetime | ModelDateTime,
    variables: Mapping[str, OutputVariable],
) -> dict[str, OutputVariable]:
    """Return coordinates used by a Veros period-average dataset."""

    coordinate_variables = {
        VEROS_TIME_DIM: time_coordinate_variable(output_time, time_dim=VEROS_TIME_DIM)
    }
    for dim in used_dimension_names(variables, excluded_dims=(VEROS_TIME_DIM,)):
        coordinate_variables[dim] = _extract_coordinate_variable(veros_state, dim)
    return coordinate_variables


def record_veros_period_output(
    adapter: ComponentOutputAdapter,
    veros_state: Any,
    *,
    output_variables: Sequence[str],
    output_time: datetime | ModelDateTime,
    dt: timedelta,
    output_frequency: str | None,
    logger: LoggerLike | None = None,
) -> bool:
    """Record one Veros snapshot and write a period average when due."""

    variables = extract_veros_output_snapshot(veros_state, output_variables)

    def build_coordinate_variables(
        mean_variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        return veros_average_coordinate_variables(
            veros_state=veros_state,
            output_time=output_time,
            variables=mean_variables,
        )

    return adapter.record_period_average_if_due(
        variables,
        time=output_time,
        dt=dt,
        output_frequency=output_frequency,
        output=lambda time: f"veros.averages.{time.strftime('%Y-%m-%d')}.nc",
        build_coordinate_variables=build_coordinate_variables,
        logger=logger,
    )


def write_veros_snapshot_output(
    state: Any,
    component_state: Any,
    output: Path,
    output_time: datetime | ModelDateTime,
    logger: LoggerLike | None = None,
) -> None:
    """Write one final Veros native-state snapshot through the shared adapter."""

    _ = component_state
    output_variables = getattr(state, "output_variables", ())
    if not output_variables:
        return

    state.output_adapter.record_snapshot(
        extract_veros_output_snapshot(state._veros_state, output_variables),
        time=output_time,
    )

    def build_coordinate_variables(
        snapshot_variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        return veros_average_coordinate_variables(
            veros_state=state._veros_state,
            output_time=output_time,
            variables=snapshot_variables,
        )

    state.output_adapter.write_snapshot(
        str(output),
        build_coordinate_variables=build_coordinate_variables,
        logger=logger,
    )


__all__ = [
    "VEROS_AVERAGE_EMPTY_ERROR_MESSAGE",
    "VEROS_TIME_DIM",
    "extract_veros_output_snapshot",
    "normalize_veros_output_variables",
    "record_veros_period_output",
    "veros_average_coordinate_variables",
    "veros_average_value_dims",
    "write_veros_snapshot_output",
]
