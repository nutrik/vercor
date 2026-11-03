"""JAX-backed streaming accumulators for period-average output variables."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.dtypes import as_jax_real_array, jax_index_dtype
from vercor._host_arrays import host_int64_array
from vercor._numerical_safety import _require_zero
from vercor._averaging import finite_sum_and_count
from vercor.output import OutputVariable, PeriodOutput

TIME_NAME = "time"
_TIME_UNITS = "microseconds since 0001-01-01 00:00:00.000000"
_MICROSECONDS_PER_SECOND = 1_000_000
_SECONDS_PER_DAY = 86_400


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
    period: PeriodOutput | None,
    *,
    time: datetime | ModelDateTime,
    dt: timedelta,
) -> bool:
    """Return whether an output average should be written for this step."""

    if period is None:
        return False

    frequency = period.frequency
    if frequency == "step":
        return True

    return is_period_end(
        time=time,
        dt=dt,
        frequency=cast(Literal["day", "month", "year"], frequency),
    )


def output_time_value_and_attrs(
    time: datetime | ModelDateTime,
) -> tuple[Any, dict[str, Any]]:
    """Return NetCDF time-coordinate values and calendar attrs for period output."""

    if isinstance(time, datetime):
        delta = time - datetime(1, 1, 1)
        return (
            host_int64_array([_timedelta_to_microseconds(delta)]),
            {
                "units": _TIME_UNITS,
                "calendar": "proleptic_gregorian",
                "isoformat": time.isoformat(),
                "day_of_year": time.timetuple().tm_yday,
            },
        )

    origin = type(time)(1, 1, 1, 0, 0, 0, 0, 1)
    model_delta = time - origin
    if not isinstance(model_delta, timedelta):
        raise TypeError("model-calendar output time subtraction must return timedelta")

    calendar = "360_day" if time.fixed_30_day_months else "noleap"
    return (
        host_int64_array([_timedelta_to_microseconds(model_delta)]),
        {
            "units": _TIME_UNITS,
            "calendar": calendar,
            "isoformat": time.isoformat(),
            "day_of_year": time.day_of_year,
            "days_per_year": time.days_per_year,
            "fixed_30_day_months": int(time.fixed_30_day_months),
        },
    )


def _timedelta_to_microseconds(delta: timedelta) -> int:
    return (
        delta.days * _SECONDS_PER_DAY + delta.seconds
    ) * _MICROSECONDS_PER_SECOND + delta.microseconds


def period_mean_sample_to_output_variable(
    sample: OutputVariable,
    *,
    time_dim: str,
    value_dims: Sequence[str] | None = None,
    dimension_order: Sequence[str] | None = None,
) -> OutputVariable:
    """Convert one period-mean sample into a one-step output variable."""

    values = as_jax_real_array(sample.values)[jnp.newaxis, ...]
    base_dims = (time_dim, *sample.dims)
    output_dims = (
        time_dim,
        *(tuple(value_dims) if value_dims is not None else sample.dims),
    )

    _validate_output_dims(base_dims, output_dims)

    if dimension_order is not None:
        ordered_prefix = tuple(dim for dim in dimension_order if dim in output_dims)
        ordered_rest = tuple(dim for dim in output_dims if dim not in ordered_prefix)
        output_dims = ordered_prefix + ordered_rest

    axes = tuple(base_dims.index(dim) for dim in output_dims)
    if axes != tuple(range(len(axes))):
        values = jnp.transpose(values, axes=axes)
    return OutputVariable(dims=output_dims, values=values, attrs=dict(sample.attrs))


def _validate_output_dims(
    base_dims: tuple[str, ...],
    output_dims: tuple[str, ...],
) -> None:
    if len(set(base_dims)) != len(base_dims):
        raise ValueError("Period average output dimensions must be unique.")
    if len(set(output_dims)) != len(output_dims) or set(output_dims) != set(base_dims):
        raise ValueError(
            "Period average output value_dims must be a permutation of sample dimensions."
        )


def _sample_sum_and_counts(
    name: str,
    sample: OutputVariable,
    *,
    summation_dim: str | None,
    reduction_counts: Any | None = None,
) -> tuple[tuple[str, ...], jax.Array, jax.Array]:
    dims = sample.dims
    if not isinstance(dims, tuple) or not all(isinstance(dim, str) for dim in dims):
        raise ValueError(f"Period average variable {name!r} has invalid dimensions.")

    try:
        values = as_jax_real_array(sample.values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Period average variable {name!r} must contain numeric values."
        ) from exc

    if values.ndim != len(dims):
        raise ValueError(
            f"Period average variable {name!r} has shape {values.shape} "
            f"but dimensions {dims}."
        )

    if reduction_counts is not None:
        counts = jnp.asarray(reduction_counts)
        if counts.shape != values.shape:
            raise ValueError(
                f"Reduction counts for {name!r} must have the same shape as sums."
            )
        if not jnp.issubdtype(counts.dtype, jnp.integer):
            raise TypeError(f"Reduction counts for {name!r} must be integer arrays.")
        _require_zero(
            jnp.count_nonzero(counts < 0),
            f"Output variable {name!r}",
            "nonnegative reduction counts",
        )
        _require_zero(
            jnp.count_nonzero((counts > 0) & ~jnp.isfinite(values)),
            f"Output variable {name!r}",
            "finite reduced sums",
        )
        return (
            dims,
            jnp.where(counts > 0, values, 0.0),
            counts.astype(jax_index_dtype()),
        )

    if summation_dim is None:
        sum_values, counts = finite_sum_and_count(values)
        return dims, sum_values, counts

    if dims.count(summation_dim) != 1:
        raise ValueError(
            f"Period average variable {name!r} must include one "
            f"{summation_dim!r} dimension."
        )
    axis = dims.index(summation_dim)
    reduced_dims = dims[:axis] + dims[axis + 1 :]  # noqa: E203
    sum_values, counts = finite_sum_and_count(values, axis=axis)
    return reduced_dims, sum_values, counts


__all__ = [
    "TIME_NAME",
    "is_period_end",
    "output_time_value_and_attrs",
    "period_mean_sample_to_output_variable",
    "should_write_period_output",
]
