from __future__ import annotations

from collections.abc import Mapping

import h5netcdf
import jax.numpy as jnp
import numpy as np

from vercor.dtypes import as_jax_real_array
from vercor.types import RuntimeArray


def read_forcing(
    data_files: Mapping[str, str],
    variable: str,
    where: str,
    flip_y: bool = False,
    *,
    mapping_name: str = "data_files",
) -> RuntimeArray:
    """Read one variable from configured NetCDF forcing files."""

    try:
        with h5netcdf.File(data_files[where], "r") as infile:
            var_obj = as_jax_real_array(np.array(infile.variables[variable]).T)
            if flip_y:
                return jnp.flip(var_obj, axis=1)
            return var_obj
    except KeyError as exc:
        raise KeyError(
            f"Provided 'where' key '{where}' not found in {mapping_name}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Error reading variable '{variable}' from forcing file '{data_files[where]}'"
        ) from exc
