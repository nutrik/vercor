from __future__ import annotations

from datetime import datetime
from typing import cast

from vercor.calendar import (
    CalendarDate as _CalendarDate,
    DAYS_PER_MONTH_GREGORIAN_LEAP as _DAYS_PER_MONTH_GREGORIAN_LEAP,
    DAYS_PER_MONTH_GREGORIAN_NO_LEAP as _DAYS_PER_MONTH_GREGORIAN_NO_LEAP,
    YearType as _YearType,
    day_of_year_from_month_day as _day_of_year_from_month_day,
    is_leap_year as _is_leap_year,
)

__all__ = [
    "daily_forcing_day_of_year",
    "daily_forcing_index",
    "day_of_year_360_to_gregorian",
    "gregorian_month_lengths",
    "noleap_day_of_year",
]


def gregorian_month_lengths(year: int, *, no_leap: bool) -> tuple[int, ...]:
    """Return Gregorian month lengths for forcing-index selection."""

    if no_leap or not _is_leap_year(year):
        return _DAYS_PER_MONTH_GREGORIAN_NO_LEAP
    return _DAYS_PER_MONTH_GREGORIAN_LEAP


def day_of_year_360_to_gregorian(
    time: _CalendarDate,
    *,
    no_leap: bool,
) -> int:
    """Map a 360-day calendar date to a Gregorian day-of-year."""

    month_lengths = gregorian_month_lengths(time.year, no_leap=no_leap)
    month_length = month_lengths[time.month - 1]
    mapped_day_in_month = ((time.day - 1) * (month_length - 1)) // 29 + 1
    return _day_of_year_from_month_day(
        month_lengths,
        time.month,
        mapped_day_in_month,
    )


def noleap_day_of_year(time: _CalendarDate) -> int:
    """Return a one-based no-leap model-calendar day-of-year."""

    if time.day_of_year is None:
        raise ValueError("ModelDateTime.day_of_year is not initialized")
    return time.day_of_year


def _validate_year_type(year_type: str | _YearType) -> _YearType:
    try:
        return _YearType(year_type)
    except ValueError as exc:
        raise ValueError("year_type must be one of: 'leap', 'noleap', '360'") from exc


def daily_forcing_day_of_year(
    time: datetime | _CalendarDate,
    *,
    year_type: str,
    no_leap: bool = True,
) -> int:
    """Return the one-based day-of-year used for daily forcing lookup."""

    validated_year_type = _validate_year_type(year_type)
    if validated_year_type is _YearType.DAY_360:
        return day_of_year_360_to_gregorian(cast(_CalendarDate, time), no_leap=no_leap)

    if validated_year_type is _YearType.GREGORIAN_NO_LEAP and not isinstance(
        time, datetime
    ):
        return noleap_day_of_year(cast(_CalendarDate, time))

    day_of_year = time.timetuple().tm_yday
    if no_leap and _is_leap_year(time.year) and day_of_year > 59:
        day_of_year -= 1
    return day_of_year


def daily_forcing_index(
    time: datetime | _CalendarDate,
    *,
    year_type: str,
    no_leap: bool = True,
) -> int:
    """Return the zero-based daily forcing index for a runtime timestamp."""

    return (
        daily_forcing_day_of_year(
            time,
            year_type=year_type,
            no_leap=no_leap,
        )
        - 1
    )
