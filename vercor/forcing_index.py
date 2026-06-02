from __future__ import annotations

from datetime import datetime
from typing import cast

from vercor.calendar import (
    CalendarDate,
    DAYS_PER_MONTH_GREGORIAN_LEAP,
    DAYS_PER_MONTH_GREGORIAN_NO_LEAP,
    day_of_year_from_month_day,
    is_leap_year,
)


def gregorian_month_lengths(year: int, *, no_leap: bool) -> tuple[int, ...]:
    """Return Gregorian month lengths for forcing-index selection."""

    if no_leap or not is_leap_year(year):
        return DAYS_PER_MONTH_GREGORIAN_NO_LEAP
    return DAYS_PER_MONTH_GREGORIAN_LEAP


def day_of_year_360_to_gregorian(
    time: CalendarDate,
    *,
    no_leap: bool,
) -> int:
    """Map a 360-day calendar date to a Gregorian day-of-year."""

    month_lengths = gregorian_month_lengths(time.year, no_leap=no_leap)
    month_length = month_lengths[time.month - 1]
    mapped_day_in_month = ((time.day - 1) * (month_length - 1)) // 29 + 1
    return day_of_year_from_month_day(
        month_lengths,
        time.month,
        mapped_day_in_month,
    )


def noleap_day_of_year(time: CalendarDate) -> int:
    """Return a one-based no-leap model-calendar day-of-year."""

    if time.day_of_year is None:
        raise ValueError("ModelDateTime.day_of_year is not initialized")
    return time.day_of_year


def daily_forcing_day_of_year(
    time: datetime | CalendarDate,
    *,
    year_type: str,
    no_leap: bool = True,
) -> int:
    """Return the one-based day-of-year used for daily forcing lookup."""

    if year_type == "360":
        return day_of_year_360_to_gregorian(cast(CalendarDate, time), no_leap=no_leap)

    if year_type == "noleap":
        return noleap_day_of_year(cast(CalendarDate, time))

    day_of_year = time.timetuple().tm_yday
    if no_leap and is_leap_year(time.year) and day_of_year > 59:
        day_of_year -= 1
    return day_of_year


def daily_forcing_index(
    time: datetime | CalendarDate,
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
