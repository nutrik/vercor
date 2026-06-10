from __future__ import annotations

from collections.abc import Mapping

import h5netcdf
import jax.numpy as jnp
import numpy as np

from vercor.dtypes import as_jax_real_array
from vercor.types import RuntimeArray


def _resolve_forcing_path(
    data_files: Mapping[str, str],
    where: str,
    mapping_name: str,
) -> str:
    try:
        return data_files[where]
    except KeyError as exc:
        raise KeyError(
            f"Provided 'where' key '{where}' not found in {mapping_name}"
        ) from exc


def _read_netcdf_variable(path: str, variable: str) -> np.ndarray:
    with h5netcdf.File(path, "r") as infile:
        try:
            return np.array(infile.variables[variable])
        except KeyError as exc:
            raise KeyError(
                f"Variable '{variable}' not found in forcing file '{path}'"
            ) from exc


def read_forcing(
    data_files: Mapping[str, str],
    variable: str,
    where: str,
    flip_y: bool = False,
    *,
    mapping_name: str = "data_files",
) -> RuntimeArray:
    """Read one variable from configured NetCDF forcing files."""

    path = _resolve_forcing_path(data_files, where, mapping_name)
    try:
        values = _read_netcdf_variable(path, variable)
    except KeyError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Error reading variable '{variable}' from forcing file '{path}'"
        ) from exc

    var_obj = as_jax_real_array(values.T)
    if flip_y:
        return jnp.flip(var_obj, axis=1)
    return var_obj
