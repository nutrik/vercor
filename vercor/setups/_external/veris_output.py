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
from collections.abc import Mapping

from vercor.types import RuntimeArray

import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.grids import RectilinearGrid
from vercor.grid_geometry import centers_to_edges
from vercor.output import (
    OutputContext,
    OutputFrame,
    OutputProvider,
    OutputVariable,
    SnapshotContext,
)
from vercor.output._dataset import time_coordinate_variable, used_dimension_names
from vercor.output._netcdf import write_netcdf_dataset
from vercor.output._period import period_mean_sample_to_output_variable

_STATE_VARIABLES = ("Area", "hIceMean", "hSnowMean", "TSurf", "uIce", "vIce")


def _sample(
    payload: Any,
    grid: RectilinearGrid,
    time: datetime | ModelDateTime,
    canonical_fields: Mapping[str, RuntimeArray],
) -> OutputFrame:
    """Build one frame from current payload arrays and native registry metadata."""
    if payload is None:
        raise ValueError("Veris output requires a native runtime payload")
    from veris.model.variables import VARIABLES
    from veris.model.diagnostics import DIAGNOSTICS

    variables = {}
    for name, definition in {
        **{name: VARIABLES[name] for name in _STATE_VARIABLES},
        **DIAGNOSTICS,
    }.items():
        native = payload.state if name in _STATE_VARIABLES else payload.diagnostics
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
        variables[name] = OutputVariable(
            dims,
            canonical_fields[name],
            {"units": units, "long_name": description, "cell_methods": "time: mean"},
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
    coordinates = {"time": time_coordinate_variable(time)}
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
    """Stateless native provider; no retained model or output cursor."""

    def sample(self, context: OutputContext) -> OutputFrame:
        """Sample current State/Diagnostics only when the coordinator requests it."""
        grid = (
            context.state.grid
            if context.state.grid is not None
            else context.component.grid
        )
        return _sample(context.payload, grid, context.time, context.state.fields())


def veris_output_provider() -> OutputProvider:
    """Return a provider compatible with shared empty/subset variable selection."""
    return _VerisOutputProvider()


def write_veris_snapshot_output(context: SnapshotContext) -> None:
    """Write the final native frame through the existing NetCDF boundary."""
    grid = (
        context.state.grid if context.state.grid is not None else context.component.grid
    )
    frame = _sample(context.payload, grid, context.time, context.state.fields())
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
