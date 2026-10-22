"""Veros period-output extraction and coordinate helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.dtypes import as_jax_index_array, as_jax_real_array
from vercor.output import JaxOutputContext, OutputContext, OutputFrame, SnapshotContext
from vercor.output._dataset import time_coordinate_variable, used_dimension_names
from vercor.output._period import TIME_NAME, period_mean_sample_to_output_variable
from vercor.output._netcdf import write_netcdf_dataset
from vercor.output import OutputVariable
from vercor.types import RuntimeArray
from veros import variables as veros_variables

VEROS_TIME_DIM = TIME_NAME
_TIMESTEP_DIM = "timesteps"
_GHOST_DIMS = ("xt", "yt", "xu", "yu")
_DEFAULT_OUTPUT_VARIABLES = (
    "temp",
    "salt",
    "u",
    "v",
    "w",
    "surface_taux",
    "surface_tauy",
    "psi",
)


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


def _current_timestep_index(vs: Any) -> RuntimeArray:
    return as_jax_index_array(vs.tau)


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
    variable = getattr(veros_state, "var_meta", {}).get(name)
    if variable is None:
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


def veros_average_coordinate_variables(
    *,
    veros_state: Any,
    output_time: datetime | ModelDateTime,
    variables: Mapping[str, OutputVariable],
) -> dict[str, OutputVariable]:
    """Return coordinates used by a Veros period-average dataset."""

    return {
        VEROS_TIME_DIM: time_coordinate_variable(output_time, time_dim=VEROS_TIME_DIM),
        **_veros_numeric_coordinates(veros_state, variables),
    }


def _veros_numeric_coordinates(
    veros_state: Any,
    variables: Mapping[str, OutputVariable],
) -> dict[str, OutputVariable]:
    """Extract native coordinate arrays without host calendar conversion."""

    coordinate_variables = {}
    for dim in used_dimension_names(variables, excluded_dims=(VEROS_TIME_DIM,)):
        coordinate_variables[dim] = _extract_coordinate_variable(veros_state, dim)
    return coordinate_variables


class _VerosOutputProvider:
    """Adapt the current payload-owned Veros state for host or JAX output."""

    def sample(self, context: OutputContext) -> OutputFrame:
        """Extract active native variables and the real post-step timestamp."""

        frame = self._sample_frame(context.payload)
        return replace(
            frame,
            coordinates={
                VEROS_TIME_DIM: time_coordinate_variable(
                    context.time,
                    time_dim=VEROS_TIME_DIM,
                ),
                **frame.coordinates,
            },
        )

    def sample_jax(self, context: JaxOutputContext) -> OutputFrame:
        """Purely extract the current native time slot for compiled means."""

        return self._sample_frame(context.payload)

    def _sample_frame(self, veros_state: Any) -> OutputFrame:
        """Share native state extraction independently of host timestamps."""

        if veros_state is None:
            raise ValueError("Veros output requires a native runtime payload.")
        native_variables = _native_output_variables(
            veros_state,
            _active_output_variable_names(veros_state),
        )
        return OutputFrame(
            native_variables,
            coordinates=_veros_numeric_coordinates(veros_state, native_variables),
            time_dimension=VEROS_TIME_DIM,
        )


def veros_output_provider() -> _VerosOutputProvider:
    """Return the native Veros provider installed by the setup factory."""

    return _VerosOutputProvider()


def _coordinate_dimension_is_extractable(veros_state: Any, dim: str) -> bool:
    if dim == _TIMESTEP_DIM:
        return True
    variable = veros_variables.VARIABLES.get(dim)
    return bool(
        variable is not None
        and _resolved_dims(variable, veros_state.settings, dim) == (dim,)
        and hasattr(veros_state.variables, dim)
    )


def _active_output_variable_names(veros_state: Any) -> tuple[str, ...]:
    """Return active native variables in Veros manifest order."""

    metadata = getattr(veros_state, "var_meta", None)
    if metadata is None:
        return _DEFAULT_OUTPUT_VARIABLES
    active_metadata = {
        name: variable
        for name, variable in metadata.items()
        if name in veros_variables.VARIABLES
        and bool(_resolve_metadata(variable.active, veros_state.settings))
        and hasattr(veros_state.variables, name)
    }
    dimensions_by_name = {
        name: _resolved_dims(variable, veros_state.settings, name)
        for name, variable in active_metadata.items()
    }
    coordinate_names = {VEROS_TIME_DIM} | {
        dim for dims in dimensions_by_name.values() for dim in dims
    }
    return tuple(
        name
        for name, dims in dimensions_by_name.items()
        if name not in coordinate_names
        and len(set(dims)) == len(dims)
        and all(_coordinate_dimension_is_extractable(veros_state, dim) for dim in dims)
    )


def _native_output_variables(
    veros_state: Any,
    variables: Sequence[str],
) -> dict[str, OutputVariable]:
    """Extract and transpose selected Veros variables for NetCDF output."""

    extracted = extract_veros_output_snapshot(veros_state, variables)
    return {
        name: OutputVariable(
            tuple(reversed(variable.dims)),
            (
                jnp.transpose(variable.values)
                if len(variable.dims) > 1
                else variable.values
            ),
            variable.attrs,
        )
        for name, variable in extracted.items()
    }


def write_veros_snapshot_output(
    context: SnapshotContext,
    *,
    variables: Sequence[str] = _DEFAULT_OUTPUT_VARIABLES,
) -> None:
    """Write one final Veros native-state snapshot through the shared adapter."""

    veros_state = context.payload
    if veros_state is None:
        raise ValueError("Veros snapshot output requires a native runtime payload.")
    selected = tuple(variables) or _DEFAULT_OUTPUT_VARIABLES
    native_variables = _native_output_variables(veros_state, selected)
    frame = OutputFrame(
        native_variables,
        coordinates=veros_average_coordinate_variables(
            veros_state=veros_state,
            output_time=context.time,
            variables=native_variables,
        ),
        time_dimension=VEROS_TIME_DIM,
    )
    write_netcdf_dataset(
        output=str(context.output_path),
        coordinate_variables=frame.coordinates,
        data_variables={
            name: period_mean_sample_to_output_variable(
                variable,
                time_dim=frame.time_dimension,
            )
            for name, variable in frame.variables.items()
        },
        global_attrs=frame.metadata or None,
        logger=context.logger,
    )


__all__ = [
    "VEROS_TIME_DIM",
    "extract_veros_output_snapshot",
    "normalize_veros_output_variables",
    "veros_average_coordinate_variables",
    "veros_output_provider",
    "write_veros_snapshot_output",
]
