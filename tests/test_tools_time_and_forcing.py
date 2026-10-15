from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import importlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import jax.numpy as jnp
import pytest
import vercor._runtime.time as runtime_time_module
import vercor.assets as assets_module
import vercor.setups._data.assets as setup_assets_module

from tests.conftest import SelectFastCases
from vercor.calendar import (
    DateTime360,
    DateTime365,
    YearType,
    is_leap_year,
    model_year_seconds,
    year_type_for_calendar,
)
from vercor._runtime.time import runtime_step_info_from_times
from vercor.exceptions import AssetError
from vercor.setups._data.assets import get_forcing_data
from vercor.time_selection import (
    datetime_to_seconds_in_year,
    get_periodic_interval,
)


@dataclass(frozen=True)
class DailyForcingIndexCase:
    case_id: str
    time: datetime | DateTime360 | DateTime365
    year_type: str
    no_leap: bool
    expected_day_of_year: int
    expected_index: int


@pytest.mark.fast_always
def test_forcing_index_resolves_daily_forcing_calendar_cases(
    select_fast_cases: SelectFastCases,
) -> None:
    forcing_index_module = importlib.import_module("vercor.forcing_index")
    cases = [
        DailyForcingIndexCase(
            case_id="gregorian-common",
            time=datetime(2001, 12, 31),
            year_type="leap",
            no_leap=False,
            expected_day_of_year=365,
            expected_index=364,
        ),
        DailyForcingIndexCase(
            case_id="gregorian-leap-day-collapses-for-noleap-forcing",
            time=datetime(2000, 2, 29),
            year_type="leap",
            no_leap=True,
            expected_day_of_year=59,
            expected_index=58,
        ),
        DailyForcingIndexCase(
            case_id="noleap-model-calendar",
            time=DateTime365(2001, 3, 1, 0, 0, 0, 0, 60),
            year_type="noleap",
            no_leap=True,
            expected_day_of_year=60,
            expected_index=59,
        ),
        DailyForcingIndexCase(
            case_id="360-model-calendar-noleap-forcing",
            time=DateTime360(2001, 2, 30, 0, 0, 0, 0, 60),
            year_type="360",
            no_leap=True,
            expected_day_of_year=59,
            expected_index=58,
        ),
        DailyForcingIndexCase(
            case_id="360-model-calendar-leap-forcing",
            time=DateTime360(2000, 2, 30, 0, 0, 0, 0, 60),
            year_type="360",
            no_leap=False,
            expected_day_of_year=60,
            expected_index=59,
        ),
    ]

    for case in select_fast_cases(
        cases, case_id=lambda case: case.case_id, min_cases=3
    ):
        owner_day = forcing_index_module.daily_forcing_day_of_year(
            case.time,
            year_type=case.year_type,
            no_leap=case.no_leap,
        )
        owner_index = forcing_index_module.daily_forcing_index(
            case.time,
            year_type=case.year_type,
            no_leap=case.no_leap,
        )

        assert owner_day == case.expected_day_of_year
        assert owner_index == case.expected_index

    assert (
        forcing_index_module.daily_forcing_index(
            datetime(2001, 1, 1),
            year_type=YearType.GREGORIAN_LEAP,
        )
        == 0
    )


@pytest.mark.fast_always
def test_forcing_index_rejects_unknown_year_type() -> None:
    forcing_index_module = importlib.import_module("vercor.forcing_index")
    time = datetime(2001, 1, 1)

    with pytest.raises(ValueError, match="year_type must be one of"):
        forcing_index_module.daily_forcing_day_of_year(time, year_type="gregorian")
    with pytest.raises(ValueError, match="year_type must be one of"):
        forcing_index_module.daily_forcing_index(time, year_type="gregorian")


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


@pytest.mark.fast_always
@pytest.mark.parametrize("month", range(12))
def test_get_periodic_interval_monthly_midpoints(month: int) -> None:
    """Monthly means belong at the middle of each record interval."""

    month_length = 30 * 86_400.0
    (n1, f1), (n2, f2) = get_periodic_interval(
        (month + 0.5) * month_length, 12 * month_length, month_length, 12
    )

    assert isinstance(n1, int)
    assert isinstance(n2, int)
    assert (n1, n2) == (month, (month + 1) % 12)
    assert (f1, f2) == (1.0, 0.0)
    records = np.arange(12, dtype=float)
    assert f1 * records[n1] + f2 * records[n2] == month


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("current_day", "expected"),
    (
        (-15.0, ((11, 1.0), (0, 0.0))),
        (0.0, ((11, 0.5), (0, 0.5))),
        (7.5, ((11, 0.25), (0, 0.75))),
        (22.5, ((0, 0.75), (1, 0.25))),
        (30.0, ((0, 0.5), (1, 0.5))),
        (352.5, ((11, 0.75), (0, 0.25))),
        (360.0, ((11, 0.5), (0, 0.5))),
        (375.0, ((0, 1.0), (1, 0.0))),
    ),
)
def test_get_periodic_interval_monthly_weights(
    current_day: float,
    expected: tuple[tuple[int, float], tuple[int, float]],
) -> None:
    """Interpolate centered records continuously across the cycle boundary."""

    day_length = 86_400.0
    actual = get_periodic_interval(
        current_day * day_length, 360 * day_length, 30 * day_length, 12
    )

    for (index, weight), (expected_index, expected_weight) in zip(
        actual, expected, strict=True
    ):
        assert isinstance(index, int)
        assert index == expected_index
        assert weight == pytest.approx(expected_weight)


@pytest.mark.parametrize("cycle", (-2, 0, 2))
@pytest.mark.parametrize("rec_spacing", (1.0, 6.0))
def test_get_periodic_interval_arbitrary_record_spacing(
    cycle: int, rec_spacing: float
) -> None:
    """Record centering applies to any spacing and repeated cycle."""

    (n1, f1), (n2, f2) = get_periodic_interval(
        (4 * cycle + 1.25) * rec_spacing, 4 * rec_spacing, rec_spacing, 4
    )

    assert (n1, n2) == (0, 1)
    assert (f1, f2) == pytest.approx((0.25, 0.75))


def test_get_periodic_interval_single_record() -> None:
    (n1, f1), (n2, f2) = get_periodic_interval(0.0, 10.0, 10.0, 1)

    assert (n1, n2) == (0, 0)
    assert f1 + f2 == pytest.approx(1.0)


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


@pytest.mark.fast_always
def test_calendar_owns_canonical_year_types_and_durations() -> None:
    assert tuple(YearType) == (
        YearType.GREGORIAN_LEAP,
        YearType.GREGORIAN_NO_LEAP,
        YearType.DAY_360,
    )
    assert YearType.GREGORIAN_LEAP == "leap"
    assert YearType.GREGORIAN_NO_LEAP == "noleap"
    assert YearType.DAY_360 == "360"
    assert model_year_seconds(YearType.GREGORIAN_LEAP) == 366 * 86_400.0
    assert model_year_seconds(YearType.GREGORIAN_NO_LEAP) == 365 * 86_400.0
    assert model_year_seconds(YearType.DAY_360) == 360 * 86_400.0


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("calendar", "year", "expected"),
    (
        ("gregorian", 2000, YearType.GREGORIAN_LEAP),
        ("gregorian", 1900, YearType.GREGORIAN_NO_LEAP),
        ("gregorian", 2001, YearType.GREGORIAN_NO_LEAP),
        ("noleap", 2000, YearType.GREGORIAN_NO_LEAP),
        ("360_day", 2000, YearType.DAY_360),
    ),
)
def test_calendar_resolves_year_type_from_existing_clock_policy(
    calendar: str,
    year: int,
    expected: YearType,
) -> None:
    assert year_type_for_calendar(calendar, year) is expected


@pytest.mark.fast_always
def test_calendar_year_helpers_reject_foreign_policy_values() -> None:
    with pytest.raises(TypeError, match="year_type must be a YearType"):
        model_year_seconds(cast(Any, "leap"))
    with pytest.raises(ValueError, match="calendar must be one of"):
        year_type_for_calendar("julian", 2000)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("time", "calendar", "year_type"),
    (
        (datetime(2000, 7, 2, 12), "gregorian", YearType.GREGORIAN_LEAP),
        (datetime(2001, 7, 2, 12), "gregorian", YearType.GREGORIAN_NO_LEAP),
        (
            DateTime365(2000, 7, 2, 12, 0, 0, 0, 183),
            "noleap",
            YearType.GREGORIAN_NO_LEAP,
        ),
        (
            DateTime360(2000, 7, 2, 12, 0, 0, 0, 182),
            "360_day",
            YearType.DAY_360,
        ),
    ),
)
def test_runtime_monthly_metadata_uses_timestamp_calendar_duration(
    time: datetime | DateTime360 | DateTime365,
    calendar: str,
    year_type: YearType,
) -> None:
    info = runtime_step_info_from_times(
        [time],
        calendar=calendar,
    )
    expected_left, expected_right = get_periodic_interval(
        current_time=datetime_to_seconds_in_year(time),
        cycle_length=model_year_seconds(year_type),
        rec_spacing=model_year_seconds(year_type) / 12.0,
        n_rec=12,
    )

    assert int(info.monthly_index_left[0]) == expected_left[0]
    assert int(info.monthly_index_right[0]) == expected_right[0]
    assert jnp.isclose(info.monthly_weight_left[0], expected_left[1])
    assert jnp.isclose(info.monthly_weight_right[0], expected_right[1])


@pytest.mark.fast_always
def test_runtime_metadata_resolves_gregorian_year_duration_per_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_year_type = runtime_time_module.year_type_for_calendar
    resolve_year_seconds = runtime_time_module.model_year_seconds
    resolved_years: list[int] = []
    resolved_year_types: list[YearType] = []

    def traced_year_type_for_calendar(calendar: str, year: int) -> YearType:
        resolved_years.append(year)
        return resolve_year_type(calendar, year)

    def traced_model_year_seconds(year_type: YearType) -> float:
        resolved_year_types.append(year_type)
        return resolve_year_seconds(year_type)

    monkeypatch.setattr(
        runtime_time_module,
        "year_type_for_calendar",
        traced_year_type_for_calendar,
    )
    monkeypatch.setattr(
        runtime_time_module,
        "model_year_seconds",
        traced_model_year_seconds,
    )
    times = [datetime(2000, 12, 31), datetime(2001, 1, 1)]

    info = runtime_step_info_from_times(times, calendar="gregorian")

    expected_intervals = [
        get_periodic_interval(
            current_time=datetime_to_seconds_in_year(time),
            cycle_length=cycle_length,
            rec_spacing=cycle_length / 12.0,
            n_rec=12,
        )
        for time, cycle_length in zip(
            times,
            (366 * 86_400.0, 365 * 86_400.0),
            strict=True,
        )
    ]
    assert resolved_years == [2000, 2001]
    assert resolved_year_types == [
        YearType.GREGORIAN_LEAP,
        YearType.GREGORIAN_NO_LEAP,
    ]
    assert info.monthly_index_left.tolist() == [
        interval[0][0] for interval in expected_intervals
    ]
    assert info.monthly_index_right.tolist() == [
        interval[1][0] for interval in expected_intervals
    ]
    assert np.allclose(
        info.monthly_weight_left,
        [interval[0][1] for interval in expected_intervals],
    )
    assert np.allclose(
        info.monthly_weight_right,
        [interval[1][1] for interval in expected_intervals],
    )
    assert info.daily_index.tolist() == [364, 0]


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
