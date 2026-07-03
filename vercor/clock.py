from dataclasses import dataclass
from datetime import datetime, timedelta
from math import floor
from typing import Iterator, Literal

import vercor.calendar as _calendar

__all__ = ["Clock"]


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

        if self.year_type in ("noleap", "360"):
            self._datetime_class: (
                type[_calendar.DateTime365] | type[_calendar.DateTime360]
            )
            if self.year_type == "noleap":
                self._datetime_class = _calendar.DateTime365
            else:
                self._datetime_class = _calendar.DateTime360

            self._start_day_of_year = self._day_of_year_for_start(self.start)
            self._start_day_index = self._start_day_of_year - 1
            self._start_seconds_of_day = (
                self.start.hour * 3600
                + self.start.minute * 60
                + self.start.second
                + self.start.microsecond / 1_000_000.0
            )

    def _day_of_year_for_start(self, start: datetime) -> int:
        if self.year_type == "360":
            if start.day > 30:
                raise ValueError(
                    "for year_type='360', start day must be between 1 and 30"
                )
            return (start.month - 1) * 30 + start.day

        if start.month == 2 and start.day == 29:
            raise ValueError("for year_type='noleap', start date cannot be February 29")

        return _calendar.DateTime365._day_of_year_from_month_day(start.month, start.day)

    def _iter_gregorian(self) -> Iterator[tuple[int, datetime, timedelta]]:
        """Iterator over Gregorian datetimes anchored at `start`."""
        time = self.start
        dt = timedelta(seconds=self.dt_seconds)
        for n in range(self.steps):
            yield n, time, dt
            time += dt

    def _iter_model_calendar(
        self,
    ) -> Iterator[tuple[int, _calendar.ModelDateTime, timedelta]]:
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

    def iter(
        self,
    ) -> Iterator[tuple[int, datetime | _calendar.ModelDateTime, timedelta]]:
        """Iterator over simulation steps using the configured stepping strategy."""
        if self.year_type == "leap":
            yield from self._iter_gregorian()
            return
        yield from self._iter_model_calendar()
