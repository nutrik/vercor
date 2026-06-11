"""Shared h5netcdf output writer for period-average datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import h5netcdf

from vercor.host_arrays import array_to_host
from vercor.output.variables import OutputVariable


def write_netcdf_dataset(
    *,
    output: str,
    coordinate_variables: Mapping[str, OutputVariable],
    data_variables: Mapping[str, OutputVariable],
    global_attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write coordinate and data variables to one NetCDF file."""

    with h5netcdf.File(output, "w") as outfile:
        for attr_name, attr_value in (
            global_attrs or {"Conventions": "CF-1.8"}
        ).items():
            if attr_value is not None:
                outfile.attrs[attr_name] = attr_value

        for name, variable in coordinate_variables.items():
            _ensure_dimensions(outfile, variable)
            output_variable = outfile.create_variable(
                name,
                variable.dims,
                data=array_to_host(variable.values),
            )
            _write_attrs(output_variable.attrs, variable.attrs)

        for name, variable in data_variables.items():
            _ensure_dimensions(outfile, variable)
            output_variable = outfile.create_variable(
                name,
                variable.dims,
                data=array_to_host(variable.values),
            )
            _write_attrs(output_variable.attrs, variable.attrs)


def _ensure_dimensions(outfile: h5netcdf.File, variable: OutputVariable) -> None:
    for dim, size in zip(variable.dims, variable.values.shape):
        if dim not in outfile.dimensions:
            outfile.dimensions[dim] = size
            continue
        existing_size = len(outfile.dimensions[dim])
        if existing_size != size:
            raise ValueError(
                f"NetCDF dimension {dim!r} has existing size {existing_size} "
                f"but new size {size}."
            )


def _write_attrs(target_attrs: Any, attrs: Mapping[str, Any]) -> None:
    for attr_name, attr_value in attrs.items():
        if attr_value is not None:
            target_attrs[attr_name] = attr_value


__all__ = ["write_netcdf_dataset"]
