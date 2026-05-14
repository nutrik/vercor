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


class ComponentForcingData:
    """Read named forcing variables from configured NetCDF files."""

    def __init__(self) -> None:
        self.DATA_FILES: dict[str, str] = {}

    def _read_forcing(
        self, variable: str, where: str, flip_y: bool = False
    ) -> RuntimeArray:
        """Read a variable from one configured forcing file as a JAX array."""

        return read_forcing(
            self.DATA_FILES,
            variable,
            where,
            flip_y=flip_y,
            mapping_name="DATA_FILES",
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"└── Forcing files: {self.DATA_FILES if self.DATA_FILES else 'No files assigned'}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(DATA_FILES={self.DATA_FILES})"
