from __future__ import annotations

from datetime import datetime
import time as _time
from typing import Protocol, cast

DAYS_PER_MONTH_GREGORIAN_LEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
DAYS_PER_MONTH_GREGORIAN_NO_LEAP = (
    31,
    28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
)
DAYS_PER_MONTH_360 = (30,) * 12


class CalendarDate(Protocol):
    @property
    def year(self) -> int: ...

    @property
    def month(self) -> int: ...

    @property
    def day(self) -> int: ...

    @property
    def day_of_year(self) -> int | None: ...

    def timetuple(self) -> _time.struct_time: ...


def is_leap_year(year: int) -> bool:
    """Return whether ``year`` is a Gregorian leap year."""

    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def day_of_year_from_month_day(
    month_lengths: tuple[int, ...],
    month: int,
    day: int,
) -> int:
    """Return one-based day-of-year for a month/day pair."""

    return sum(month_lengths[: month - 1]) + day


def month_day_from_day_of_year(
    month_lengths: tuple[int, ...],
    day_of_year: int,
) -> tuple[int, int]:
    """Return ``(month, day)`` for one-based day-of-year."""

    day_cursor = day_of_year
    for month, month_len in enumerate(month_lengths, start=1):
        if day_cursor <= month_len:
            return month, day_cursor
        day_cursor -= month_len
    raise ValueError(f"invalid day_of_year={day_of_year}")


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
