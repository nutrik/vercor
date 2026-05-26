"""JAXGCM output cadence and NetCDF writing helpers."""

from __future__ import annotations

from collections.abc import MutableSequence
from datetime import datetime, timedelta
from typing import Any, Literal, cast

import xarray as xr

from vercor.calendar import ModelDateTime
from vercor.jax_logging import LoggerLike, get_default_logger


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
    """Return whether JAXGCM should write an output average for this step."""

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


def write_jax_gcm_averages_output(
    predictions: MutableSequence[Any],
    output: str,
    logger: LoggerLike | None = None,
) -> None:
    """Write mean JAXGCM predictions to NetCDF and clear the prediction buffer."""

    ds = cast(
        xr.Dataset,
        xr.merge([prediction.to_xarray() for prediction in predictions]),
    )

    log = logger if logger is not None else get_default_logger()
    log.info(f"Output file: {output:s}")

    t_end = ds.time.isel(time=-1)
    ds.mean(dim="time", keep_attrs=True, keepdims=True).assign_coords(
        time=[t_end.values]
    ).transpose("time", "wvi_id", "hsg_level", "level", "lat", "lon").to_netcdf(
        output, engine="h5netcdf"
    )

    predictions.clear()
