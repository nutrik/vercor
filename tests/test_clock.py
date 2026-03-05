from datetime import datetime, timedelta

import pytest

from vercor.clock import Clock, DateTime360, DateTime365


def test_datetime365_str_matches_datetime_str_without_microseconds() -> None:
    model_time = DateTime365(
        year=2025,
        month=1,
        day=2,
        hour=3,
        minute=4,
        second=5,
        microsecond=0,
        day_of_year=2,
    )

    reference = datetime(2025, 1, 2, 3, 4, 5)
    assert str(model_time) == str(reference)


def test_datetime360_str_matches_datetime_str_with_microseconds() -> None:
    model_time = DateTime360(
        year=2025,
        month=1,
        day=2,
        hour=3,
        minute=4,
        second=5,
        microsecond=123456,
        day_of_year=2,
    )

    reference = datetime(2025, 1, 2, 3, 4, 5, 123456)
    assert str(model_time) == str(reference)


def test_model_datetime_repr_is_constructor_style() -> None:
    model_time = DateTime360(
        year=2026,
        month=12,
        day=30,
        hour=6,
        minute=7,
        second=8,
        microsecond=900000,
        day_of_year=360,
    )

    assert repr(model_time) == "DateTime360(2026, 12, 30, 6, 7, 8, 900000, 360)"


def test_model_datetime_strftime_matches_datetime_for_common_directives() -> None:
    model_time = DateTime365(
        year=2025,
        month=1,
        day=2,
        hour=3,
        minute=4,
        second=5,
        microsecond=123456,
        day_of_year=2,
    )
    reference = datetime(2025, 1, 2, 3, 4, 5, 123456)
    fmt = "%Y-%m-%d %H:%M:%S.%f"

    assert model_time.strftime(fmt) == reference.strftime(fmt)


def test_model_datetime_strftime_uses_model_day_of_year_in_360_calendar() -> None:
    model_time = DateTime360(
        year=2026,
        month=12,
        day=30,
        hour=6,
        minute=7,
        second=8,
        microsecond=900000,
        day_of_year=360,
    )

    assert model_time.strftime("%j") == "360"


def test_model_datetime_strftime_handles_literal_percent_f() -> None:
    model_time = DateTime365(
        year=2025,
        month=1,
        day=2,
        hour=3,
        minute=4,
        second=5,
        microsecond=123456,
        day_of_year=2,
    )

    assert model_time.strftime("%%f %f") == "%f 123456"


def test_model_datetime_add_timedelta_wraps_360_day_year() -> None:
    model_time = DateTime360(
        year=2025,
        month=12,
        day=30,
        hour=6,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=360,
    )

    out = model_time + timedelta(days=1)
    assert (out.year, out.month, out.day, out.day_of_year, out.hour) == (
        2026,
        1,
        1,
        1,
        6,
    )


def test_model_datetime_radd_timedelta() -> None:
    model_time = DateTime360(
        year=2025,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=1,
    )

    out = timedelta(hours=12) + model_time
    assert (out.year, out.month, out.day, out.hour) == (2025, 1, 1, 12)


def test_model_datetime_subtract_timedelta() -> None:
    model_time = DateTime360(
        year=2025,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=1,
    )

    out = model_time - timedelta(seconds=1)
    assert isinstance(out, DateTime360)
    assert (
        out.year,
        out.month,
        out.day,
        out.day_of_year,
        out.hour,
        out.minute,
        out.second,
    ) == (
        2024,
        12,
        30,
        360,
        23,
        59,
        59,
    )


def test_model_datetime_subtract_model_datetime_returns_timedelta() -> None:
    earlier = DateTime360(
        year=2025,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=1,
    )
    later = DateTime360(
        year=2025,
        month=1,
        day=2,
        hour=1,
        minute=2,
        second=3,
        microsecond=400000,
        day_of_year=2,
    )

    diff = later - earlier
    assert diff == timedelta(days=1, hours=1, minutes=2, seconds=3, microseconds=400000)


def test_model_datetime_comparisons() -> None:
    t0 = DateTime365(
        year=2025,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=1,
    )
    t1 = DateTime365(
        year=2025,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=1,
        microsecond=0,
        day_of_year=1,
    )

    assert t0 < t1
    assert t0 <= t1
    assert t1 > t0
    assert t1 >= t0
    assert t0 != t1
    assert t0 == t0


def test_model_datetime_mixed_calendar_arithmetic_and_order_raise() -> None:
    t360 = DateTime360(
        year=2025,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=1,
    )
    t365 = DateTime365(
        year=2025,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=1,
    )

    with pytest.raises(TypeError, match="different calendars"):
        _ = t360 - t365

    with pytest.raises(TypeError, match="different calendars"):
        _ = t360 < t365

    assert (t360 == t365) is False


def test_alternative_clock_360_day_calendar_wraps_month_and_year() -> None:
    clock = Clock(
        start=datetime(2025, 12, 30, 6, 0, 0),
        dt_seconds=86400.0,
        steps=3,
        year_type="360",
    )

    values = list(clock.iter())

    _, t0, _ = values[0]
    _, t1, _ = values[1]
    _, t2, _ = values[2]

    assert isinstance(t0, DateTime360)
    assert isinstance(t1, DateTime360)
    assert isinstance(t2, DateTime360)
    assert (t0.year, t0.month, t0.day, t0.day_of_year) == (2025, 12, 30, 360)
    assert (t1.year, t1.month, t1.day, t1.day_of_year) == (2026, 1, 1, 1)
    assert (t2.year, t2.month, t2.day, t2.day_of_year) == (2026, 1, 2, 2)


def test_alternative_clock_noleap_calendar_skips_feb_29() -> None:
    clock = Clock(
        start=datetime(2024, 2, 28, 0, 0, 0),
        dt_seconds=86400.0,
        steps=3,
        year_type="noleap",
    )

    values = list(clock.iter())

    _, t0, _ = values[0]
    _, t1, _ = values[1]
    _, t2, _ = values[2]

    assert isinstance(t0, DateTime365)
    assert isinstance(t1, DateTime365)
    assert isinstance(t2, DateTime365)
    assert t0 == DateTime365(2024, 2, 28, 0, 0, 0, 0, 59)
    assert t1 == DateTime365(2024, 3, 1, 0, 0, 0, 0, 60)
    assert t2 == DateTime365(2024, 3, 2, 0, 0, 0, 0, 61)


def test_alternative_clock_iter_is_gregorian_when_not_fixed_30_day_months() -> None:
    clock = Clock(
        start=datetime(2025, 1, 30, 12, 0, 0),
        dt_seconds=86400.0,
        steps=2,
        year_type="noleap",
    )

    values = list(clock.iter())

    assert values[0][1] == DateTime365(2025, 1, 30, 12, 0, 0, 0, 30)
    assert values[1][1] == DateTime365(2025, 1, 31, 12, 0, 0, 0, 31)


def test_clock_leap_uses_standard_datetime_and_keeps_feb_29() -> None:
    clock = Clock(
        start=datetime(2024, 2, 29), dt_seconds=86400.0, steps=2, year_type="leap"
    )

    values = list(clock.iter())
    assert isinstance(values[0][1], datetime)
    assert values[0][1] == datetime(2024, 2, 29, 0, 0, 0)
    assert values[1][1] == datetime(2024, 3, 1, 0, 0, 0)


def test_clock_noleap_rejects_feb_29_start() -> None:
    with pytest.raises(ValueError, match="cannot be February 29"):
        Clock(
            start=datetime(2024, 2, 29),
            dt_seconds=3600.0,
            steps=1,
            year_type="noleap",
        )


def test_clock_noleap_100_year_daily_run_reaches_year_100() -> None:
    clock = Clock(
        start=datetime(2000, 1, 3, 0, 0, 0),
        dt_seconds=86400.0,
        steps=365 * 100 - 2,
        year_type="noleap",
    )

    values = list(clock.iter())
    _, last_time, _ = values[-1]

    assert isinstance(last_time, DateTime365)
    assert last_time.year - 2000 + 1 == 100
