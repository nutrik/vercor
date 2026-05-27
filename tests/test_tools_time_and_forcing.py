from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vercor.assets as assets_module
import vercor.setups.data.assets as setup_assets_module

from tests._tools_support import make_coupler
from tests.assertions import assert_allclose_compact
from tests.conftest import SelectFastCases
from vercor.calendar import DateTime360, DateTime365
from vercor.exceptions import AssetError
from vercor.calendar import is_leap_year
from vercor.setups.data.assets import get_forcing_data
from vercor.time_selection import (
    datetime_to_seconds_in_year,
    get_field_at_specific_time,
    get_field_time_slice,
    get_periodic_interval,
)


@dataclass(frozen=True)
class TimeSliceCase:
    case_id: str
    data: np.ndarray
    time: datetime | DateTime360 | DateTime365
    no_leap: bool
    expected: np.ndarray | float


@pytest.mark.fast_always
def test_get_field_time_slice_cases(
    select_fast_cases: SelectFastCases,
) -> None:
    cases = [
        TimeSliceCase(
            case_id="gregorian-start",
            data=np.arange(365 * 2, dtype=float).reshape(365, 2),
            time=datetime(2001, 1, 1),
            no_leap=False,
            expected=np.array([0.0, 1.0]),
        ),
        TimeSliceCase(
            case_id="gregorian-end",
            data=np.arange(365 * 2, dtype=float).reshape(365, 2),
            time=datetime(2001, 12, 31),
            no_leap=False,
            expected=np.array([728.0, 729.0]),
        ),
        TimeSliceCase(
            case_id="gregorian-noleap-feb29",
            data=np.arange(366, dtype=float),
            time=datetime(2000, 2, 29),
            no_leap=True,
            expected=58.0,
        ),
        TimeSliceCase(
            case_id="gregorian-leap-feb29",
            data=np.arange(366, dtype=float),
            time=datetime(2000, 2, 29),
            no_leap=False,
            expected=59.0,
        ),
        TimeSliceCase(
            case_id="360-january-map",
            data=np.arange(365, dtype=float),
            time=DateTime360(2001, 1, 30, 0, 0, 0, 0, 30),
            no_leap=True,
            expected=30.0,
        ),
        TimeSliceCase(
            case_id="360-february-map",
            data=np.arange(365, dtype=float),
            time=DateTime360(2001, 2, 3, 0, 0, 0, 0, 33),
            no_leap=True,
            expected=32.0,
        ),
        TimeSliceCase(
            case_id="360-february-noleap",
            data=np.arange(365, dtype=float),
            time=DateTime360(2001, 2, 30, 0, 0, 0, 0, 60),
            no_leap=True,
            expected=58.0,
        ),
        TimeSliceCase(
            case_id="360-february-leap",
            data=np.arange(366, dtype=float),
            time=DateTime360(2000, 2, 30, 0, 0, 0, 0, 60),
            no_leap=False,
            expected=59.0,
        ),
        TimeSliceCase(
            case_id="365-dayofyear",
            data=np.arange(365, dtype=float),
            time=DateTime365(2001, 3, 1, 0, 0, 0, 0, 60),
            no_leap=True,
            expected=59.0,
        ),
    ]

    for case in select_fast_cases(
        cases, case_id=lambda case: case.case_id, min_cases=2
    ):
        out = get_field_time_slice(
            "foo", {"foo": case.data}, case.time, no_leap=case.no_leap
        )
        assert isinstance(out, jax.Array)
        assert_allclose_compact(out, case.expected, label=case.case_id)


def test_get_field_time_slice_returns_jax_array_for_jax_backed_data() -> None:
    data = {"foo": jnp.arange(365 * 2, dtype=jnp.float64).reshape(365, 2)}

    out = get_field_time_slice("foo", data, datetime(2001, 1, 2), no_leap=False)

    assert isinstance(out, jax.Array)
    assert_allclose_compact(out, np.asarray([2.0, 3.0]))


def test_get_field_at_specific_time_weights_and_interpolation() -> None:
    coupler = make_coupler(year_in_seconds=12.0)

    lat, lon, nrec = 2, 3, 12
    arr = np.zeros((nrec, lat, lon), dtype=float)
    arr[0, ...] = 0.0
    arr[1, ...] = 10.0
    data = {"foo": arr}

    current_time = coupler.clock.start + timedelta(seconds=0.25)
    total_seconds = (current_time - coupler.clock.start).total_seconds()
    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=total_seconds,
        cycle_length=coupler.settings.year_in_seconds,
        rec_spacing=coupler.settings.year_in_seconds / 12.0,
        n_rec=12,
    )

    assert isinstance(n1, int)
    assert isinstance(n2, int)
    assert n1 == 0
    assert n2 == 1
    assert np.isclose(f1 + f2, 1.0)

    out = get_field_at_specific_time("foo", data, coupler, current_time=current_time)
    assert isinstance(out, jax.Array)
    assert_allclose_compact(out, 2.5)


def test_get_field_at_specific_time_boundary_record() -> None:
    coupler = make_coupler(year_in_seconds=120.0)
    arr = np.zeros((12, 2, 2), dtype=float)
    arr[1, ...] = 7.0

    rec_spacing = coupler.settings.year_in_seconds / 12.0
    current_time = coupler.clock.start + timedelta(seconds=rec_spacing)

    out = get_field_at_specific_time(
        "foo", {"foo": arr}, coupler, current_time=current_time
    )
    assert_allclose_compact(out, 7.0)


def test_get_field_at_specific_time_accepts_jax_backed_forcing_cube() -> None:
    coupler = make_coupler(year_in_seconds=12.0)
    arr = jnp.zeros((12, 2, 3), dtype=jnp.float64)
    arr = arr.at[0].set(jnp.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]))

    out = get_field_at_specific_time(
        "foo", {"foo": arr}, coupler, current_time=coupler.clock.start
    )

    assert isinstance(out, jax.Array)
    assert out.shape == (2, 3)
    assert_allclose_compact(out, np.asarray(arr[0]))


def test_get_field_at_specific_time_axis_ordering() -> None:
    coupler = make_coupler(year_in_seconds=12.0)
    arr = np.zeros((12, 2, 3), dtype=float)
    arr[0] = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])

    out = get_field_at_specific_time(
        "foo", {"foo": arr}, coupler, current_time=coupler.clock.start
    )
    expected = arr[0]

    assert out.shape == (2, 3)
    assert_allclose_compact(out, expected)


def test_get_field_at_specific_time_uses_coupler_clock_start_when_time_is_none() -> (
    None
):
    coupler = make_coupler(year_in_seconds=12.0)
    arr = np.zeros((12, 2, 2), dtype=float)
    arr[0, ...] = 3.0

    out = get_field_at_specific_time("foo", {"foo": arr}, coupler, current_time=None)
    assert_allclose_compact(out, 3.0)


def test_get_field_at_specific_time_wraps_across_year_end() -> None:
    coupler = make_coupler(year_in_seconds=12.0)
    arr = np.zeros((12, 2, 2), dtype=float)
    arr[11, ...] = 100.0
    arr[0, ...] = 20.0

    current_time = coupler.clock.start + timedelta(seconds=11.75)
    out = get_field_at_specific_time("foo", {"foo": arr}, coupler, current_time)

    assert_allclose_compact(out, 40.0)


def test_datetime_to_seconds_in_year_for_datetime() -> None:
    dt = datetime(2001, 2, 3, 4, 5, 6, 700000)
    expected = 33 * 86400 + 4 * 3600 + 5 * 60 + 6 + 0.7

    assert np.isclose(datetime_to_seconds_in_year(dt), expected)


def test_datetime_to_seconds_in_year_for_model_datetime_with_arithmetic() -> None:
    base = DateTime360(2001, 1, 1, 0, 0, 0, 0, 1)
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

    assert isinstance(n1, int)
    assert isinstance(n2, int)
    assert n1 == 1
    assert n2 == 2
    assert np.isclose(f1 + f2, 1.0)


def test_get_periodic_interval_exact_last_record_boundary_wraps_to_first() -> None:
    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=11.0,
        cycle_length=12.0,
        rec_spacing=1.0,
        n_rec=12,
    )

    assert n1 == 11
    assert n2 == 0
    assert np.isclose(f1, 1.0)
    assert np.isclose(f2, 0.0)


def test_get_periodic_interval_exact_cycle_boundary_resets_to_first_record() -> None:
    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=12.0,
        cycle_length=12.0,
        rec_spacing=1.0,
        n_rec=12,
    )

    assert n1 == 0
    assert n2 == 1
    assert np.isclose(f1, 1.0)
    assert np.isclose(f2, 0.0)


@pytest.mark.fast_always
def test_is_leap_year_cases(select_fast_cases: SelectFastCases) -> None:
    cases: list[tuple[str, int, bool]] = [
        ("divisible-by-400", 2000, True),
        ("century-not-leap", 1900, False),
        ("ordinary-leap", 2004, True),
        ("ordinary-common", 2001, False),
    ]

    for _case_id, year, expected in select_fast_cases(
        cases, case_id=lambda case: case[0], min_cases=2
    ):
        assert is_leap_year(year) is expected


def test_get_forcing_data_valid_and_invalid_file_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_name = "model.nc"
    surface_name = "surface.nc"
    model_bytes = b"model-level-data"
    surface_bytes = b"surface-data"

    (tmp_path / model_name).write_bytes(model_bytes)
    (tmp_path / surface_name).write_bytes(surface_bytes)

    monkeypatch.setattr(assets_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        setup_assets_module,
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
