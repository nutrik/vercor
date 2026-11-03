"""Sample native sea ice through VerCOR's shared host output coordinator.

Native arrays include two halos in (x, y) order. Samples remove halos and
transpose dimensions together to (y, x). U/V locations use native west/south
faces; no interpolation or thickness/Area renormalization is applied.
hIceMean and hSnowMean remain volume per total grid-cell area (m). Diagnostic
salt forcing has local virtual units (g/kg)*m/s; actual ocean mass/salt inputs
are separate downward-positive SI rates in the native diagnostic registry.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from collections.abc import Callable, Mapping

from vercor.types import RuntimeArray

import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.grids import RectilinearGrid
from vercor.grid_geometry import centers_to_edges
from vercor.output import (
    JaxOutputContext,
    JaxOutputProvider,
    OutputContext,
    OutputFrame,
    OutputVariable,
    SnapshotContext,
)
from vercor.output._dataset import time_coordinate_variable, used_dimension_names
from vercor.output._netcdf import write_netcdf_dataset
from vercor.output._period import period_mean_sample_to_output_variable
from vercor.output._selection import component_output_names, with_runtime_fields


def _native_names(payload: Any, fields: Mapping[str, RuntimeArray]) -> tuple[str, ...]:
    """Enumerate initialized native registries and maintained derived fields."""
    from veris.model.variables import VARIABLES
    from veris.model.diagnostics import DIAGNOSTICS

    return (
        *(name for name in VARIABLES if hasattr(payload.state, name)),
        *(name for name in DIAGNOSTICS if hasattr(payload.diagnostics, name)),
        *(
            name
            for name in (
                "sea_surface_heat_flux",
                "sea_surface_mass_flux",
                "sea_surface_salt_mass_flux",
                "ocean_u_stress",
                "ocean_v_stress",
            )
            if name in fields
        ),
        "ice_surface_temperature",
        "ice_surface_albedo",
    )


def _sample(
    payload: Any,
    grid: RectilinearGrid,
    time: datetime | ModelDateTime | None,
    canonical_fields: Mapping[str, RuntimeArray],
    surface_state_fields: Callable[[Any], Mapping[str, RuntimeArray]],
    selected: tuple[str, ...] | None = None,
) -> OutputFrame:
    """Build one frame from current payload arrays and native registry metadata."""
    if payload is None:
        raise ValueError("Veris output requires a native runtime payload")
    from veris.model.variables import VARIABLES
    from veris.model.diagnostics import DIAGNOSTICS

    variables = {}
    definitions = {**VARIABLES, **DIAGNOSTICS}
    names = _native_names(payload, canonical_fields) if selected is None else selected
    endpoint_fields = (
        surface_state_fields(payload) if "ice_surface_albedo" in names else {}
    )
    for name in names:
        if name not in definitions:
            continue
        definition = definitions[name]
        native = payload.state if name in VARIABLES else payload.diagnostics
        variables[name] = OutputVariable(
            tuple(reversed(definition.dimensions)),
            jnp.asarray(getattr(native, name))[2:-2, 2:-2].T,
            definition.netcdf_attributes(),
        )
    for name, dims, units, description in (
        (
            "sea_surface_heat_flux",
            ("y_center", "x_center"),
            "W m-2",
            "Downward total ocean heat input",
        ),
        (
            "sea_surface_mass_flux",
            ("y_center", "x_center"),
            "kg m-2 s-1",
            "Downward total ocean mass input",
        ),
        (
            "sea_surface_salt_mass_flux",
            ("y_center", "x_center"),
            "kg m-2 s-1",
            "Downward actual ocean salt mass input",
        ),
        (
            "ocean_u_stress",
            ("y_center", "x_east_face"),
            "N m-2",
            "Eastward total ocean surface stress",
        ),
        (
            "ocean_v_stress",
            ("y_north_face", "x_center"),
            "N m-2",
            "Northward total ocean surface stress",
        ),
    ):
        if name not in names:
            continue
        variables[name] = OutputVariable(
            dims,
            canonical_fields[name],
            {"units": units, "long_name": description, "cell_methods": "time: mean"},
        )
    for name, units, description in (
        ("ice_surface_temperature", "K", "Current per-ice-area surface temperature"),
        ("ice_surface_albedo", "1", "Current per-ice-area surface albedo"),
    ):
        if name not in names:
            continue
        variables[name] = OutputVariable(
            ("y_center", "x_center"),
            (
                payload.state.TSurf[2:-2, 2:-2].T
                if name == "ice_surface_temperature"
                else endpoint_fields[name]
            ),
            {"units": units, "long_name": description, "cell_methods": "time: point"},
        )
    x_edges = grid.longitude_edges
    y_edges = grid.latitude_edges
    values = {
        "x_center": grid.longitude,
        "y_center": grid.latitude,
        "x_face": (
            centers_to_edges(grid.longitude, "lon") if x_edges is None else x_edges
        )[:-1],
        "y_face": (
            centers_to_edges(grid.latitude, "lat") if y_edges is None else y_edges
        )[:-1],
        "x_east_face": (
            centers_to_edges(grid.longitude, "lon") if x_edges is None else x_edges
        )[1:],
        "y_north_face": (
            centers_to_edges(grid.latitude, "lat") if y_edges is None else y_edges
        )[1:],
    }
    coordinates = {} if time is None else {"time": time_coordinate_variable(time)}
    for name in used_dimension_names(variables):
        longitude = name.startswith("x_")
        coordinates[name] = OutputVariable(
            (name,),
            values[name],
            {
                "units": "degrees_east" if longitude else "degrees_north",
                "standard_name": "longitude" if longitude else "latitude",
            },
        )
    return OutputFrame(variables, coordinates=coordinates, metadata={"model": "Veris"})


class _VerisOutputProvider:
    """Native provider retaining only a setup-bound endpoint diagnostic function."""

    def __init__(
        self, surface_state_fields: Callable[[Any], Mapping[str, RuntimeArray]]
    ) -> None:
        self._surface_state_fields = surface_state_fields

    def sample(self, context: OutputContext) -> OutputFrame:
        """Sample selected native and canonical fields with a host timestamp."""
        return self._sample_frame(context, context.time)

    def sample_jax(self, context: JaxOutputContext) -> OutputFrame:
        """Purely extract selected fields without host calendar values."""
        return self._sample_frame(context, None)

    def _sample_frame(
        self,
        context: OutputContext | JaxOutputContext,
        time: datetime | ModelDateTime | None,
    ) -> OutputFrame:
        if context.payload is None:
            raise ValueError("Veris output requires a native runtime payload")
        grid = (
            context.state.grid
            if context.state.grid is not None
            else context.component.grid
        )
        fields = context.state.fields()
        native, canonical = component_output_names(
            context, _native_names(context.payload, fields)
        )
        frame = _sample(
            context.payload, grid, time, fields, self._surface_state_fields, native
        )
        return with_runtime_fields(
            frame, context, canonical, horizontal_dims=("y_center", "x_center")
        )


def veris_output_provider(
    surface_state_fields: Callable[[Any], Mapping[str, RuntimeArray]],
) -> JaxOutputProvider:
    """Return a provider compatible with shared empty/subset variable selection."""
    return _VerisOutputProvider(surface_state_fields)


def write_veris_snapshot_output(
    surface_state_fields: Callable[[Any], Mapping[str, RuntimeArray]],
    context: SnapshotContext,
) -> None:
    """Write the final native frame through the existing NetCDF boundary."""
    grid = (
        context.state.grid if context.state.grid is not None else context.component.grid
    )
    frame = _sample(
        context.payload,
        grid,
        context.time,
        context.state.fields(),
        surface_state_fields,
    )
    write_netcdf_dataset(
        output=str(context.output_path),
        coordinate_variables=frame.coordinates,
        data_variables={
            name: period_mean_sample_to_output_variable(
                variable, time_dim=frame.time_dimension
            )
            for name, variable in frame.variables.items()
        },
        global_attrs=frame.metadata,
        logger=context.logger,
    )
