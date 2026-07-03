from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest
import vercor.assets as assets_module
import vercor.setups.data.assets as setup_assets_module

from tests.conftest import SelectFastCases
from vercor.calendar import DateTime360, DateTime365
from vercor.exceptions import AssetError
from vercor.calendar import is_leap_year
from vercor.setups.data.assets import get_forcing_data
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
