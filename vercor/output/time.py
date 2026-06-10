"""Shared period-output cadence and NetCDF time-coordinate helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal, cast

from vercor.calendar import ModelDateTime
from vercor.host_arrays import host_int64_array

TIME_NAME = "time"
_TIME_UNITS = "microseconds since 0001-01-01 00:00:00.000000"
_MICROSECONDS_PER_SECOND = 1_000_000
_SECONDS_PER_DAY = 86_400


def is_period_end(
    time: datetime | ModelDateTime,
    dt: timedelta,
    frequency: Literal["day", "month", "year"],
) -> bool:
    """Return whether ``time + dt`` crosses the requested calendar boundary."""

    next_time = time + dt

    if frequency == "day":
        return (
            next_time.year != time.year
            or next_time.month != time.month
            or next_time.day != time.day
        )
    if frequency == "month":
        return next_time.year != time.year or next_time.month != time.month

    return next_time.year != time.year


def should_write_period_output(
    *,
    time: datetime | ModelDateTime,
    dt: timedelta,
    output_frequency: str | None,
) -> bool:
    """Return whether an output average should be written for this step."""

    if output_frequency is None:
        return True

    if not isinstance(output_frequency, str):
        return False

    frequency = output_frequency.lower()
    if frequency not in ("day", "month", "year"):
        return False

    return is_period_end(
        time=time,
        dt=dt,
        frequency=cast(Literal["day", "month", "year"], frequency),
    )


def output_time_value_and_attrs(
    time: datetime | ModelDateTime,
) -> tuple[Any, dict[str, Any]]:
    """Return NetCDF time-coordinate values and calendar attrs for period output."""

    if isinstance(time, datetime):
        delta = time - datetime(1, 1, 1)
        return (
            host_int64_array([_timedelta_to_microseconds(delta)]),
            {
                "units": _TIME_UNITS,
                "calendar": "proleptic_gregorian",
                "isoformat": time.isoformat(),
                "day_of_year": time.timetuple().tm_yday,
            },
        )

    origin = type(time)(1, 1, 1, 0, 0, 0, 0, 1)
    model_delta = time - origin
    if not isinstance(model_delta, timedelta):
        raise TypeError("model-calendar output time subtraction must return timedelta")

    calendar = "360_day" if time.fixed_30_day_months else "noleap"
    return (
        host_int64_array([_timedelta_to_microseconds(model_delta)]),
        {
            "units": _TIME_UNITS,
            "calendar": calendar,
            "isoformat": time.isoformat(),
            "day_of_year": time.day_of_year,
            "days_per_year": time.days_per_year,
            "fixed_30_day_months": int(time.fixed_30_day_months),
        },
    )


def _timedelta_to_microseconds(delta: timedelta) -> int:
    return (
        delta.days * _SECONDS_PER_DAY + delta.seconds
    ) * _MICROSECONDS_PER_SECOND + delta.microseconds


__all__ = [
    "TIME_NAME",
    "is_period_end",
    "output_time_value_and_attrs",
    "should_write_period_output",
]
