from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest
import xarray as xr

from tests.assertions import assert_allclose_compact
from vercor.forcing_data import read_forcing

pytestmark = pytest.mark.fast_always


def test_read_forcing_reads_legacy_transposed_jax_array(tmp_path: Path) -> None:
    path = tmp_path / "forcing.nc"
    source = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    xr.Dataset({"foo": (("x", "y"), source)}).to_netcdf(path)

    out = read_forcing({"sample": str(path)}, "foo", "sample")

    assert isinstance(out, jax.Array)
    assert_allclose_compact(out, source.T)


def test_read_forcing_flips_legacy_latitude_axis(tmp_path: Path) -> None:
    path = tmp_path / "forcing.nc"
    source = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    xr.Dataset({"foo": (("x", "y"), source)}).to_netcdf(path)

    out = read_forcing({"sample": str(path)}, "foo", "sample", flip_y=True)

    assert isinstance(out, jax.Array)
    assert_allclose_compact(out, np.flip(source.T, axis=1))


def test_read_forcing_reports_missing_mapping_key(tmp_path: Path) -> None:
    path = tmp_path / "forcing.nc"
    xr.Dataset({"foo": (("x",), np.asarray([1.0]))}).to_netcdf(path)

    with pytest.raises(KeyError, match="Provided 'where' key 'missing'"):
        read_forcing({"sample": str(path)}, "foo", "missing")


def test_read_forcing_reports_missing_netcdf_variable(tmp_path: Path) -> None:
    path = tmp_path / "forcing.nc"
    xr.Dataset({"foo": (("x",), np.asarray([1.0]))}).to_netcdf(path)

    with pytest.raises(KeyError, match="Variable 'bar' not found"):
        read_forcing({"sample": str(path)}, "bar", "sample")


def test_read_forcing_wraps_broken_netcdf_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.nc"
    broken.write_text("not-a-netcdf-file", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Error reading variable 'foo'"):
        read_forcing({"broken": str(broken)}, "foo", "broken")
