from __future__ import annotations

from collections.abc import Mapping

from vercor.forcing_data import read_forcing as _read_forcing
from vercor.types import RuntimeArray


def read_forcing(
    data_files: Mapping[str, str],
    variable: str,
    where: str,
    flip_y: bool = False,
) -> RuntimeArray:
    """Read one variable from configured NetCDF forcing files."""

    return _read_forcing(data_files, variable, where, flip_y=flip_y)
