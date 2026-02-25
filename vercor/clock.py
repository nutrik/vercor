from dataclasses import dataclass
from datetime import datetime, timedelta
from math import floor
import time
from typing import Callable, ClassVar, Iterator, Literal, Self


_DAYS_PER_MONTH_GREGORIAN_LEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_DAYS_PER_MONTH_GREGORIAN_NO_LEAP = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MICROSECONDS_PER_DAY = 86_400_000_000


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

    def timetuple(self) -> time.struct_time:
        """Return a `datetime.timetuple()`-compatible value for this CustomDateTime."""
        if self.day_of_year is None:
            raise ValueError("day_of_year is not initialized")

        days_before_year = (self.year - 1) * self.days_per_year
        days_before_day = self.day_of_year - 1
        weekday = (days_before_year + days_before_day) % 7

        return time.struct_time(
            (
                self.year,
                self.month,
                self.day,
                self.hour,
                self.minute,
                self.second,
                weekday,
                self.day_of_year,
                -1,  # tm_isdst is not applicable for model calendar
            )
        )

    def strftime(self, fmt: str) -> str:
        """Format the datetime using `datetime.strftime`-style directives.

        Formatting is delegated to `time.strftime` with this object's
        calendar-aware `timetuple()`, which preserves model-calendar values
        such as `%j` in 360-day years. `%f` is handled explicitly to match
        `datetime.strftime` microsecond behavior.
        """
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
        out = time.strftime(processed_fmt, self.timetuple())
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
        return sum(cls._MONTH_LENGTHS[: month - 1]) + day

    @classmethod
    def _month_day_from_day_of_year(cls, day_of_year: int) -> tuple[int, int]:
        day_cursor = day_of_year
        for month, month_len in enumerate(cls._MONTH_LENGTHS, start=1):
            if day_cursor <= month_len:
                return month, day_cursor
            day_cursor -= month_len

        raise ValueError(f"invalid day_of_year={day_of_year}")

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
    DAYS_PER_YEAR: ClassVar[int] = 365
    FIXED_30_DAY_MONTHS: ClassVar[bool] = False
    _MONTH_LENGTHS: ClassVar[tuple[int, ...]] = _DAYS_PER_MONTH_GREGORIAN_NO_LEAP


@dataclass(frozen=True, repr=False)
class DateTime360(_ModelDateTimeBase):
    DAYS_PER_YEAR: ClassVar[int] = 360
    FIXED_30_DAY_MONTHS: ClassVar[bool] = True
    _MONTH_LENGTHS: ClassVar[tuple[int, ...]] = (30,) * 12


ModelDateTime = DateTime365 | DateTime360
CustomDateTime = _ModelDateTimeBase


@dataclass
class Clock:
    """Calendar-aware clock with configurable synthetic year/month structure.

    Notes:
        - `start` is a standard datetime and may be any valid Gregorian date.
                - `year_type` can be "leap", "noleap", or "360".
                - A single public `iter()` is exposed, and the internal stepping logic
                    is selected during initialization.
    """

    start: datetime
    dt_seconds: float
    steps: int
    year_type: Literal["leap", "noleap", "360"] = "leap"

    def __post_init__(self) -> None:
        if self.year_type not in ("leap", "noleap", "360"):
            raise ValueError("year_type must be one of: 'leap', 'noleap', '360'")

        if self.steps < 0:
            raise ValueError("steps must be non-negative")

        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")

        self._iter_impl: Callable[
            [],
            Iterator[tuple[int, datetime | ModelDateTime, timedelta]],
        ]

        if self.year_type in ("noleap", "360"):
            self._iter_impl = self._iter_model_calendar
            self._datetime_class: type[DateTime365] | type[DateTime360]
            if self.year_type == "noleap":
                self._datetime_class = DateTime365
            else:
                self._datetime_class = DateTime360

            self._start_day_of_year = self._day_of_year_for_start(self.start)
            self._start_day_index = self._start_day_of_year - 1
            self._start_seconds_of_day = (
                self.start.hour * 3600
                + self.start.minute * 60
                + self.start.second
                + self.start.microsecond / 1_000_000.0
            )
        else:
            self._iter_impl = self._iter_gregorian

    def _day_of_year_for_start(self, start: datetime) -> int:
        if self.year_type == "360":
            if start.day > 30:
                raise ValueError(
                    "for year_type='360', start day must be between 1 and 30"
                )
            return (start.month - 1) * 30 + start.day

        if start.month == 2 and start.day == 29:
            raise ValueError("for year_type='noleap', start date cannot be February 29")

        return DateTime365._day_of_year_from_month_day(start.month, start.day)

    @property
    def days_per_year(self) -> int | None:
        if self.year_type == "360":
            return 360
        if self.year_type == "noleap":
            return 365
        return None

    @property
    def fixed_30_day_months(self) -> bool:
        return self.year_type == "360"

    def _iter_gregorian(self) -> Iterator[tuple[int, datetime, timedelta]]:
        """Iterator over Gregorian datetimes anchored at `start`."""
        time = self.start
        dt = timedelta(seconds=self.dt_seconds)
        for n in range(self.steps):
            yield n, time, dt
            time += dt

    def _iter_model_calendar(self) -> Iterator[tuple[int, ModelDateTime, timedelta]]:
        """Iterator over simulation time steps in synthetic model calendars."""
        dt = timedelta(seconds=self.dt_seconds)
        days_per_year = self._datetime_class.DAYS_PER_YEAR

        for n in range(self.steps):
            elapsed_seconds = n * self.dt_seconds
            total_seconds = self._start_seconds_of_day + elapsed_seconds

            day_offset = floor(total_seconds / 86_400.0)
            seconds_of_day = total_seconds - day_offset * 86_400.0

            total_microseconds = int(round(seconds_of_day * 1_000_000.0))
            extra_day, micros_of_day = divmod(total_microseconds, 86_400_000_000)
            day_offset += extra_day

            hour, rem = divmod(micros_of_day, 3_600_000_000)
            minute, rem = divmod(rem, 60_000_000)
            second, microsecond = divmod(rem, 1_000_000)

            total_day_index = self._start_day_index + day_offset
            year_offset, day_index_in_year = divmod(total_day_index, days_per_year)

            year = self.start.year + year_offset
            day_of_year = day_index_in_year + 1
            month, day = self._datetime_class._month_day_from_day_of_year(day_of_year)

            yield n, self._datetime_class(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
                day_of_year=day_of_year,
            ), dt

    def iter(self) -> Iterator[tuple[int, datetime | ModelDateTime, timedelta]]:
        """Iterator over simulation steps using the configured stepping strategy."""
        yield from self._iter_impl()
