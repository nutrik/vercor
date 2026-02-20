from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import floor
import time
from typing import Callable
from typing import Literal
from typing import Iterator


_DAYS_PER_MONTH_GREGORIAN_LEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_DAYS_PER_MONTH_GREGORIAN_NO_LEAP = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MICROSECONDS_PER_DAY = 86_400_000_000


@dataclass(frozen=True)
class CustomDateTime:
    """Date-time value in a model calendar.

    This object is detached from Gregorian month lengths and supports either:
    - no-leap 365-day years with regular month lengths
    - 360-day years with 12 months of 30 days each
    """

    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    day_of_year: int
    days_per_year: Literal[360, 365]
    fixed_30_day_months: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.days_per_year not in (360, 365):
            raise ValueError("days_per_year must be either 360 or 365")

        if self.days_per_year == 360:
            object.__setattr__(self, "fixed_30_day_months", True)
        else:
            object.__setattr__(self, "fixed_30_day_months", False)

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
            f"{self.day_of_year}, {self.days_per_year}, {self.fixed_30_day_months}"
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

    def _same_calendar(self, other: "CustomDateTime") -> bool:
        return (
            self.days_per_year == other.days_per_year
            and self.fixed_30_day_months == other.fixed_30_day_months
        )

    def _day_seconds_microseconds(self) -> int:
        return (
            self.hour * 3_600_000_000
            + self.minute * 60_000_000
            + self.second * 1_000_000
            + self.microsecond
        )

    def _to_ordinal_microseconds(self) -> int:
        total_day_index = (self.year - 1) * self.days_per_year + (self.day_of_year - 1)
        return (
            total_day_index * _MICROSECONDS_PER_DAY + self._day_seconds_microseconds()
        )

    @staticmethod
    def _month_day_from_day_of_year(
        day_of_year: int,
        days_per_year: Literal[360, 365],
        fixed_30_day_months: bool,
    ) -> tuple[int, int]:
        if fixed_30_day_months:
            month = (day_of_year - 1) // 30 + 1
            day = (day_of_year - 1) % 30 + 1
            return month, day

        day_cursor = day_of_year
        for month, month_len in enumerate(_DAYS_PER_MONTH_GREGORIAN_NO_LEAP, start=1):
            if day_cursor <= month_len:
                return month, day_cursor
            day_cursor -= month_len

        raise ValueError(f"invalid day_of_year={day_of_year}")

    def _from_ordinal_microseconds(self, total_microseconds: int) -> "CustomDateTime":
        total_day_index, micros_of_day = divmod(
            total_microseconds, _MICROSECONDS_PER_DAY
        )

        year_offset, day_index_in_year = divmod(total_day_index, self.days_per_year)
        year = 1 + year_offset
        if year < 1:
            raise OverflowError("date value out of range")

        day_of_year = day_index_in_year + 1
        month, day = self._month_day_from_day_of_year(
            day_of_year,
            self.days_per_year,
            self.fixed_30_day_months,
        )

        hour, rem = divmod(micros_of_day, 3_600_000_000)
        minute, rem = divmod(rem, 60_000_000)
        second, microsecond = divmod(rem, 1_000_000)

        return CustomDateTime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond,
            day_of_year=day_of_year,
            days_per_year=self.days_per_year,
        )

    def __add__(self, other: object) -> "CustomDateTime":
        if not isinstance(other, timedelta):
            return NotImplemented

        delta_microseconds = (
            other.days * 86_400 + other.seconds
        ) * 1_000_000 + other.microseconds
        return self._from_ordinal_microseconds(
            self._to_ordinal_microseconds() + delta_microseconds
        )

    def __radd__(self, other: object) -> "CustomDateTime":
        if not isinstance(other, timedelta):
            return NotImplemented
        return self + other

    def __sub__(self, other: object) -> "CustomDateTime | timedelta":
        if isinstance(other, timedelta):
            return self + (-other)

        if isinstance(other, CustomDateTime):
            if not self._same_calendar(other):
                raise TypeError(
                    "cannot subtract CustomDateTime values with different calendars"
                )

            diff_microseconds = (
                self._to_ordinal_microseconds() - other._to_ordinal_microseconds()
            )
            return timedelta(microseconds=diff_microseconds)

        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CustomDateTime):
            return NotImplemented

        if not self._same_calendar(other):
            return False

        return self._to_ordinal_microseconds() == other._to_ordinal_microseconds()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CustomDateTime):
            return NotImplemented

        if not self._same_calendar(other):
            raise TypeError(
                "cannot compare CustomDateTime values with different calendars"
            )

        return self._to_ordinal_microseconds() < other._to_ordinal_microseconds()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, CustomDateTime):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, CustomDateTime):
            return NotImplemented
        return not self <= other

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, CustomDateTime):
            return NotImplemented
        return not self < other


@dataclass
class Clock:
    """Calendar-aware clock with configurable synthetic year/month structure.

    Notes:
        - `start` is a standard datetime and may be any valid Gregorian date.
        - `days_per_year` can be 360 or 365.
        - A single public `iter()` is exposed, and the internal stepping logic
          is selected during initialization.
    """

    start: datetime
    dt_seconds: float
    steps: int
    days_per_year: Literal[360, 365] = 365
    fixed_30_day_months: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.days_per_year not in (360, 365):
            raise ValueError("days_per_year must be either 360 or 365")

        if self.days_per_year == 360:
            self.fixed_30_day_months = True
        else:
            self.fixed_30_day_months = False

        if self.steps < 0:
            raise ValueError("steps must be non-negative")

        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")

        self._iter_impl: Callable[
            [],
            Iterator[tuple[int, datetime | CustomDateTime, timedelta]],
        ]

        if self.fixed_30_day_months:
            self._iter_impl = self._iter_custom_calendar
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
        if start.day > 30:
            raise ValueError(
                "for fixed_30_day_months=True, start day must be between 1 and 30"
            )
        return (start.month - 1) * 30 + start.day

    def _month_day_from_day_of_year(self, day_of_year: int) -> tuple[int, int]:
        month = (day_of_year - 1) // 30 + 1
        day = (day_of_year - 1) % 30 + 1
        return month, day

    def _iter_gregorian(self) -> Iterator[tuple[int, datetime, timedelta]]:
        """Iterator over Gregorian datetimes anchored at `start`."""
        time = self.start
        dt = timedelta(seconds=self.dt_seconds)
        for n in range(self.steps):
            yield n, time, dt
            time += dt

    def _iter_custom_calendar(self) -> Iterator[tuple[int, CustomDateTime, timedelta]]:
        """Iterator over simulation time steps in a fixed 360-day calendar."""
        dt = timedelta(seconds=self.dt_seconds)

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
            year_offset, day_index_in_year = divmod(total_day_index, self.days_per_year)

            year = self.start.year + year_offset
            day_of_year = day_index_in_year + 1
            month, day = self._month_day_from_day_of_year(day_of_year)

            yield n, CustomDateTime(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
                day_of_year=day_of_year,
                days_per_year=self.days_per_year,
            ), dt

    def iter(self) -> Iterator[tuple[int, datetime | CustomDateTime, timedelta]]:
        """Iterator over simulation steps using the configured stepping strategy."""
        yield from self._iter_impl()
