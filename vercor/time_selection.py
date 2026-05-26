from __future__ import annotations

from datetime import datetime
from typing import Mapping, Protocol

import jax.numpy as jnp

from vercor.clock import (
    DateTime360,
    DateTime365,
    ModelDateTime,
)
from vercor.calendar import daily_forcing_day_of_year
from vercor.types import RuntimeArray


class _ClockWithStart(Protocol):
    @property
    def start(self) -> datetime: ...


class _SettingsWithYearInSeconds(Protocol):
    @property
    def year_in_seconds(self) -> float: ...


class SupportsFieldTimeLookup(Protocol):
    @property
    def clock(self) -> _ClockWithStart: ...

    @property
    def settings(self) -> _SettingsWithYearInSeconds: ...


def get_periodic_interval(
    current_time: float, cycle_length: float, rec_spacing: float, n_rec: int
) -> tuple[tuple[int, float], tuple[int, float]]:
    """Return record indices and weights for periodic linear interpolation."""

    current_time = current_time % cycle_length
    t_idx_1 = int(current_time // rec_spacing)
    t_idx_2 = (1 + t_idx_1) % n_rec
    weight_2 = (current_time - rec_spacing * t_idx_1) / rec_spacing
    weight_1 = 1.0 - weight_2
    return (t_idx_1, weight_1), (t_idx_2, weight_2)


def datetime_to_seconds_in_year(dt: datetime | ModelDateTime) -> float:
    """Convert a model time to elapsed seconds since the start of its year."""

    if isinstance(dt, datetime):
        year_start = datetime(dt.year, 1, 1)
        return (dt - year_start).total_seconds()

    day_of_year = dt.day_of_year
    if day_of_year is None:
        raise ValueError("ModelDateTime.day_of_year is not initialized")

    return (
        (day_of_year - 1) * 86_400.0
        + dt.hour * 3_600.0
        + dt.minute * 60.0
        + dt.second
        + dt.microsecond / 1_000_000.0
    )


def _custom_360_day_to_gregorian_day_of_year(
    time: datetime | ModelDateTime,
    no_leap: bool,
) -> int:
    return daily_forcing_day_of_year(time, year_type="360", no_leap=no_leap)


def get_field_time_slice(
    field_name: str,
    data: Mapping[str, RuntimeArray],
    time: datetime | ModelDateTime,
    no_leap: bool = True,
) -> RuntimeArray:
    """Return a field indexed by day-of-year without time interpolation."""

    if isinstance(time, DateTime360):
        tm_yday = _custom_360_day_to_gregorian_day_of_year(time, no_leap=no_leap)
    elif isinstance(time, DateTime365):
        tm_yday = daily_forcing_day_of_year(
            time,
            year_type="noleap",
            no_leap=no_leap,
        )
    else:
        tm_yday = daily_forcing_day_of_year(
            time,
            year_type="leap",
            no_leap=no_leap,
        )

    time_index = tm_yday - 1
    out: RuntimeArray = jnp.asarray(data[field_name])[time_index, ...]
    return out


def get_field_at_specific_time(
    field_name: str,
    data: Mapping[str, RuntimeArray],
    coupler: SupportsFieldTimeLookup,
    current_time: datetime | ModelDateTime | None = None,
) -> RuntimeArray:
    """Return a monthly field interpolated to a specific model time."""

    total_seconds = datetime_to_seconds_in_year(
        coupler.clock.start if current_time is None else current_time
    )

    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=total_seconds,
        cycle_length=coupler.settings.year_in_seconds,
        rec_spacing=coupler.settings.year_in_seconds / 12.0,
        n_rec=12,
    )

    arr = jnp.asarray(data[field_name])
    out: RuntimeArray = f1 * arr[n1, ...] + f2 * arr[n2, ...]
    return out
