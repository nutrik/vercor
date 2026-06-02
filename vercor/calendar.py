from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import time as _time
from typing import ClassVar, Literal, Protocol, Self

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
_MICROSECONDS_PER_DAY = 86_400_000_000


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


@dataclass(frozen=True)
class _ModelDateTimeBase:
    """Base class for model-calendar datetime values."""

    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    day_of_year: int | None = None

    DAYS_PER_YEAR: ClassVar[int] = 0
    FIXED_30_DAY_MONTHS: ClassVar[bool] = False
    _MONTH_LENGTHS: ClassVar[tuple[int, ...]] = ()

    def __post_init__(self) -> None:
        if not (1 <= self.month <= 12):
            raise ValueError("month must be in 1..12")

        max_day = self._MONTH_LENGTHS[self.month - 1]
        if not (1 <= self.day <= max_day):
            raise ValueError(f"day must be in 1..{max_day} for month={self.month}")

        if not (0 <= self.hour <= 23):
            raise ValueError("hour must be in 0..23")
        if not (0 <= self.minute <= 59):
            raise ValueError("minute must be in 0..59")
        if not (0 <= self.second <= 59):
            raise ValueError("second must be in 0..59")
        if not (0 <= self.microsecond <= 999_999):
            raise ValueError("microsecond must be in 0..999999")

        computed_day_of_year = self._day_of_year_from_month_day(self.month, self.day)
        if self.day_of_year is None:
            object.__setattr__(self, "day_of_year", computed_day_of_year)
        elif self.day_of_year != computed_day_of_year:
            raise ValueError(
                "day_of_year is inconsistent with month/day for this calendar"
            )

    @property
    def days_per_year(self) -> Literal[360, 365]:
        return self.DAYS_PER_YEAR  # type: ignore[return-value]

    @property
    def fixed_30_day_months(self) -> bool:
        return self.FIXED_30_DAY_MONTHS

    def __str__(self) -> str:
        date = f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        time_part = f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}"
        if self.microsecond:
            time_part = f"{time_part}.{self.microsecond:06d}"
        return f"{date} {time_part}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.year}, {self.month}, {self.day}, "
            f"{self.hour}, {self.minute}, {self.second}, {self.microsecond}, "
            f"{self.day_of_year}"
            ")"
        )

    def isoformat(self) -> str:
        return (
            f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
            f"T{self.hour:02d}:{self.minute:02d}:{self.second:02d}"
            f".{self.microsecond:06d}"
        )

    def timetuple(self) -> _time.struct_time:
        """Return a ``datetime.timetuple()``-compatible model-calendar value."""

        if self.day_of_year is None:
            raise ValueError("day_of_year is not initialized")

        days_before_year = (self.year - 1) * self.days_per_year
        days_before_day = self.day_of_year - 1
        weekday = (days_before_year + days_before_day) % 7

        return _time.struct_time(
            (
                self.year,
                self.month,
                self.day,
                self.hour,
                self.minute,
                self.second,
                weekday,
                self.day_of_year,
                -1,
            )
        )

    def strftime(self, fmt: str) -> str:
        """Format the datetime using ``datetime.strftime``-style directives."""

        microsecond_token = "__VERCOR_CUSTOMDATETIME_MICROSECOND__"

        processed_fmt_parts: list[str] = []
        i = 0
        while i < len(fmt):
            char = fmt[i]
            if char != "%" or i + 1 >= len(fmt):
                processed_fmt_parts.append(char)
                i += 1
                continue

            directive = fmt[i + 1]
            if directive == "f":
                processed_fmt_parts.append(microsecond_token)
            else:
                processed_fmt_parts.append(f"%{directive}")
            i += 2

        processed_fmt = "".join(processed_fmt_parts)
        out = _time.strftime(processed_fmt, self.timetuple())
        return out.replace(microsecond_token, f"{self.microsecond:06d}")

    def _same_calendar(self, other: "_ModelDateTimeBase") -> bool:
        return type(self) is type(other)

    def _day_seconds_microseconds(self) -> int:
        return (
            self.hour * 3_600_000_000
            + self.minute * 60_000_000
            + self.second * 1_000_000
            + self.microsecond
        )

    def _to_ordinal_microseconds(self) -> int:
        if self.day_of_year is None:
            raise ValueError("day_of_year is not initialized")

        total_day_index = (self.year - 1) * self.days_per_year + (self.day_of_year - 1)
        return (
            total_day_index * _MICROSECONDS_PER_DAY + self._day_seconds_microseconds()
        )

    @classmethod
    def _day_of_year_from_month_day(cls, month: int, day: int) -> int:
        return day_of_year_from_month_day(cls._MONTH_LENGTHS, month, day)

    @classmethod
    def _month_day_from_day_of_year(cls, day_of_year: int) -> tuple[int, int]:
        return month_day_from_day_of_year(cls._MONTH_LENGTHS, day_of_year)

    def _from_ordinal_microseconds(self, total_microseconds: int) -> Self:
        total_day_index, micros_of_day = divmod(
            total_microseconds, _MICROSECONDS_PER_DAY
        )

        year_offset, day_index_in_year = divmod(total_day_index, self.days_per_year)
        year = 1 + year_offset
        if year < 1:
            raise OverflowError("date value out of range")

        day_of_year = day_index_in_year + 1
        month, day = self._month_day_from_day_of_year(day_of_year)

        hour, rem = divmod(micros_of_day, 3_600_000_000)
        minute, rem = divmod(rem, 60_000_000)
        second, microsecond = divmod(rem, 1_000_000)

        return type(self)(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond,
            day_of_year=day_of_year,
        )

    def __add__(self, other: object) -> Self:
        if not isinstance(other, timedelta):
            return NotImplemented

        delta_microseconds = (
            other.days * 86_400 + other.seconds
        ) * 1_000_000 + other.microseconds
        return self._from_ordinal_microseconds(
            self._to_ordinal_microseconds() + delta_microseconds
        )

    def __radd__(self, other: object) -> Self:
        if not isinstance(other, timedelta):
            return NotImplemented
        return self + other

    def __sub__(self, other: object) -> Self | timedelta:
        if isinstance(other, timedelta):
            return self + (-other)

        if isinstance(other, _ModelDateTimeBase):
            if not self._same_calendar(other):
                raise TypeError(
                    "cannot subtract model datetime values with different calendars"
                )

            diff_microseconds = (
                self._to_ordinal_microseconds() - other._to_ordinal_microseconds()
            )
            return timedelta(microseconds=diff_microseconds)

        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ModelDateTimeBase):
            return NotImplemented

        if not self._same_calendar(other):
            return False

        return self._to_ordinal_microseconds() == other._to_ordinal_microseconds()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ModelDateTimeBase):
            return NotImplemented

        if not self._same_calendar(other):
            raise TypeError(
                "cannot compare model datetime values with different calendars"
            )

        return self._to_ordinal_microseconds() < other._to_ordinal_microseconds()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, _ModelDateTimeBase):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, _ModelDateTimeBase):
            return NotImplemented
        return not self <= other

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, _ModelDateTimeBase):
            return NotImplemented
        return not self < other


@dataclass(frozen=True, repr=False)
class DateTime365(_ModelDateTimeBase):
    """No-leap 365-day model-calendar datetime."""

    DAYS_PER_YEAR: ClassVar[int] = 365
    FIXED_30_DAY_MONTHS: ClassVar[bool] = False
    _MONTH_LENGTHS: ClassVar[tuple[int, ...]] = DAYS_PER_MONTH_GREGORIAN_NO_LEAP


@dataclass(frozen=True, repr=False)
class DateTime360(_ModelDateTimeBase):
    """360-day model-calendar datetime with fixed 30-day months."""

    DAYS_PER_YEAR: ClassVar[int] = 360
    FIXED_30_DAY_MONTHS: ClassVar[bool] = True
    _MONTH_LENGTHS: ClassVar[tuple[int, ...]] = DAYS_PER_MONTH_360


ModelDateTime = DateTime365 | DateTime360
CustomDateTime = _ModelDateTimeBase


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
    """Return Gregorian month lengths for forcing-index selection.

    This compatibility delegate preserves the historic ``vercor.calendar``
    import path while ``vercor.forcing_index`` owns daily forcing-index policy.
    """

    from vercor.forcing_index import gregorian_month_lengths as _month_lengths

    return _month_lengths(year, no_leap=no_leap)


def day_of_year_360_to_gregorian(
    time: CalendarDate,
    *,
    no_leap: bool,
) -> int:
    """Map a 360-day calendar date to a Gregorian day-of-year.

    This compatibility delegate preserves the historic ``vercor.calendar``
    import path while ``vercor.forcing_index`` owns daily forcing-index policy.
    """

    from vercor.forcing_index import day_of_year_360_to_gregorian as _map_day

    return _map_day(time, no_leap=no_leap)


def noleap_day_of_year(time: CalendarDate) -> int:
    """Return a one-based no-leap model-calendar day-of-year.

    This compatibility delegate preserves the historic ``vercor.calendar``
    import path while ``vercor.forcing_index`` owns daily forcing-index policy.
    """

    from vercor.forcing_index import noleap_day_of_year as _noleap_day

    return _noleap_day(time)


def daily_forcing_day_of_year(
    time: datetime | CalendarDate,
    *,
    year_type: str,
    no_leap: bool = True,
) -> int:
    """Return the one-based day-of-year used for daily forcing lookup.

    This compatibility delegate preserves the historic ``vercor.calendar``
    import path while ``vercor.forcing_index`` owns daily forcing-index policy.
    """

    from vercor.forcing_index import daily_forcing_day_of_year as _day_of_year

    return _day_of_year(time, year_type=year_type, no_leap=no_leap)


def daily_forcing_index(
    time: datetime | CalendarDate,
    *,
    year_type: str,
    no_leap: bool = True,
) -> int:
    """Return the zero-based daily forcing index for a runtime timestamp.

    This compatibility delegate preserves the historic ``vercor.calendar``
    import path while ``vercor.forcing_index`` owns daily forcing-index policy.
    """

    from vercor.forcing_index import daily_forcing_index as _daily_index

    return _daily_index(time, year_type=year_type, no_leap=no_leap)
