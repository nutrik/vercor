from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import vercor.tools as tools_module

from vercor.clock import Clock, DateTime360, DateTime365
from vercor.exceptions import AssetError, CouplerError
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.tools import (
    _append_unique,
    _flatten_fields,
    datetime_to_seconds_in_year,
    get_component,
    get_forcing_data,
    get_field_at_specific_time,
    get_field_time_slice,
    get_periodic_interval,
    grids_identical,
    is_leap_year,
)


@dataclass
class DummyCoupler:
    clock: Clock
    settings: VercorSettings


def make_coupler(year_in_seconds: float) -> DummyCoupler:
    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=1.0, steps=1)
    settings = VercorSettings(year_in_seconds=year_in_seconds)
    return DummyCoupler(clock=clock, settings=settings)


@dataclass
class DummyComponentA:
    name: str = "a"


@dataclass
class DummyComponentB:
    name: str = "b"


def test_get_field_at_specific_time_weights_and_interpolation() -> None:
    coupler = make_coupler(year_in_seconds=12.0)

    # Build a (lat, lon, time) array with constant values per record
    lat, lon, nrec = 2, 3, 12
    arr = np.zeros((lat, lon, nrec), dtype=float)
    arr[..., 0] = 0.0
    arr[..., 1] = 10.0
    data = {"foo": arr}

    current_time = coupler.clock.start + timedelta(seconds=0.25)

    total_seconds = (current_time - coupler.clock.start).total_seconds()
    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=total_seconds,
        cycle_length=coupler.settings.year_in_seconds,
        rec_spacing=coupler.settings.year_in_seconds / 12.0,
        n_rec=12,
    )

    assert n1 == 0
    assert n2 == 1
    assert np.isclose(f1 + f2, 1.0)

    out = get_field_at_specific_time("foo", data, coupler, current_time=current_time)  # type: ignore
    assert np.allclose(out, 2.5)


def test_get_field_at_specific_time_boundary_record() -> None:
    coupler = make_coupler(year_in_seconds=120.0)

    lat, lon, nrec = 2, 2, 12
    arr = np.zeros((lat, lon, nrec), dtype=float)
    arr[..., 1] = 7.0
    data = {"foo": arr}

    rec_spacing = coupler.settings.year_in_seconds / 12.0
    current_time = coupler.clock.start + timedelta(seconds=rec_spacing)

    out = get_field_at_specific_time("foo", data, coupler, current_time=current_time)  # type: ignore
    assert np.allclose(out, 7.0)


def test_get_field_at_specific_time_axis_ordering() -> None:
    coupler = make_coupler(year_in_seconds=12.0)

    # lat-major data; output is expected to be (lon, lat) due to swapaxes
    arr = np.zeros((2, 3, 12), dtype=float)
    arr[:, :, 0] = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    data = {"foo": arr}

    out = get_field_at_specific_time("foo", data, coupler, current_time=coupler.clock.start)  # type: ignore
    expected = arr[:, :, 0].swapaxes(-2, -1)

    assert out.shape == (3, 2)
    assert np.allclose(out, expected)


def test_get_field_at_specific_time_uses_coupler_clock_start_when_time_is_none() -> (
    None
):
    coupler = make_coupler(year_in_seconds=12.0)
    arr = np.zeros((2, 2, 12), dtype=float)
    arr[..., 0] = 3.0
    data = {"foo": arr}

    out = get_field_at_specific_time("foo", data, coupler, current_time=None)  # type: ignore
    assert np.allclose(out, 3.0)


def test_get_field_time_slice_basic_indexing() -> None:
    data = {"foo": np.arange(365 * 2, dtype=float).reshape(365, 2)}

    time = datetime(2001, 1, 1)
    out = get_field_time_slice("foo", data, time)
    assert np.allclose(out, data["foo"][0, :])

    time = datetime(2001, 12, 31)
    out = get_field_time_slice("foo", data, time)
    assert np.allclose(out, data["foo"][364, :])


def test_get_field_time_slice_no_leap_year_adjustment() -> None:
    data = {"foo": np.arange(366, dtype=float)}

    time = datetime(2000, 2, 29)
    out = get_field_time_slice("foo", data, time, no_leap=True)

    # Feb 29 in a leap year is mapped to Feb 28 when no_leap=True
    assert np.isclose(out, data["foo"][58])


def test_get_field_time_slice_leap_day_retained_when_requested() -> None:
    data = {"foo": np.arange(366, dtype=float)}

    time = datetime(2000, 2, 29)
    out = get_field_time_slice("foo", data, time, no_leap=False)

    assert np.isclose(out, data["foo"][59])


def test_get_field_time_slice_model_datetime_360_maps_to_real_month_lengths() -> None:
    data = {"foo": np.arange(365, dtype=float)}

    time = DateTime360(
        year=2001,
        month=1,
        day=30,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=30,
    )
    out = get_field_time_slice("foo", data, time, no_leap=True)
    assert np.isclose(out, data["foo"][30])

    time = DateTime360(
        year=2001,
        month=2,
        day=3,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=33,
    )
    out = get_field_time_slice("foo", data, time, no_leap=True)
    assert np.isclose(out, data["foo"][32])


def test_get_field_time_slice_model_datetime_360_february_non_leap() -> None:
    data = {"foo": np.arange(365, dtype=float)}

    time = DateTime360(
        year=2001,
        month=2,
        day=30,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=60,
    )
    out = get_field_time_slice("foo", data, time, no_leap=True)

    assert np.isclose(out, data["foo"][58])


def test_get_field_time_slice_model_datetime_360_february_leap_allowed() -> None:
    data = {"foo": np.arange(366, dtype=float)}

    time = DateTime360(
        year=2000,
        month=2,
        day=30,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=60,
    )
    out = get_field_time_slice("foo", data, time, no_leap=False)

    assert np.isclose(out, data["foo"][59])


def test_get_field_time_slice_model_datetime_365_uses_day_of_year_directly() -> None:
    data = {"foo": np.arange(365, dtype=float)}
    time = DateTime365(
        year=2001,
        month=3,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=60,
    )
    out = get_field_time_slice("foo", data, time, no_leap=True)
    assert np.isclose(out, data["foo"][59])


def test_datetime_to_seconds_in_year_for_model_datetime_with_arithmetic() -> None:
    base = DateTime360(
        year=2001,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=1,
    )
    shifted = base + timedelta(days=1, hours=2, minutes=3, seconds=4, microseconds=5)
    assert shifted - base == timedelta(
        days=1, hours=2, minutes=3, seconds=4, microseconds=5
    )

    seconds = datetime_to_seconds_in_year(shifted)
    assert np.isclose(seconds, 1 * 86400 + 2 * 3600 + 3 * 60 + 4 + 5e-6)


def test_get_periodic_interval_wraps_with_time_beyond_cycle() -> None:
    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=13.25,
        cycle_length=12.0,
        rec_spacing=1.0,
        n_rec=12,
    )

    assert n1 == 1
    assert n2 == 2
    assert np.isclose(f1 + f2, 1.0)


def test_is_leap_year_cases() -> None:
    assert is_leap_year(2000)
    assert not is_leap_year(1900)
    assert is_leap_year(2004)
    assert not is_leap_year(2001)


def test_get_forcing_data_valid_and_invalid_file_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_name = "model.nc"
    surface_name = "surface.nc"
    model_bytes = b"model-level-data"
    surface_bytes = b"surface-data"

    (tmp_path / model_name).write_bytes(model_bytes)
    (tmp_path / surface_name).write_bytes(surface_bytes)

    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {
            "era5_model_levels": {
                "filename": model_name,
                "md5": hashlib.md5(model_bytes).hexdigest(),
            },
            "era5_surface": {
                "filename": surface_name,
                "md5": hashlib.md5(surface_bytes).hexdigest(),
            },
        },
    )

    model_level = get_forcing_data("era5_model_levels")
    surface = get_forcing_data("era5_surface")

    assert isinstance(model_level, Path)
    assert isinstance(surface, Path)
    assert str(model_level).endswith(model_name)
    assert str(surface).endswith(surface_name)

    with pytest.raises(AssetError, match="Unknown file_type"):
        get_forcing_data("unknown")


def test_flatten_fields_and_append_unique() -> None:
    flattened = _flatten_fields(["a", ("b", "c"), "d"])
    assert flattened == ["a", "b", "c", "d"]

    target = ["a", "b"]
    _append_unique(target, ["b", "c", "d", "a"])
    assert target == ["a", "b", "c", "d"]


def test_grids_identical_detects_equal_and_unequal_grids() -> None:
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([-1.0, 0.0])
    g0 = RectilinearGrid("g0", longitude=lon, latitude=lat)
    g1 = RectilinearGrid("g1", longitude=lon.copy(), latitude=lat.copy())
    g2 = RectilinearGrid("g2", longitude=np.array([0.0, 1.5, 2.0]), latitude=lat.copy())

    assert grids_identical(g0, g1)
    assert not grids_identical(g0, g2)


def test_get_component_returns_single_and_raises_for_ambiguous_or_missing() -> None:
    allcomponents: dict[str, object] = {
        "a": DummyComponentA(name="ATM"),
        "b": DummyComponentB(name="OCN"),
    }

    selected = get_component(cast(Any, allcomponents), "ATM")
    assert isinstance(selected, DummyComponentA)

    with pytest.raises(CouplerError, match="No component"):
        get_component(cast(Any, allcomponents), "UNKNOWN")

    with pytest.raises(CouplerError, match="Multiple"):
        get_component(
            cast(
                Any,
                {
                    "a": DummyComponentA(name="OCN"),
                    "b": DummyComponentA(name="OCN"),
                },
            ),
            "OCN",
        )
