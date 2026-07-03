from __future__ import annotations

from datetime import datetime

from vercor.calendar import ModelDateTime


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
