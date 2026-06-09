from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tests.assertions import assert_allclose_compact
from vercor.setups.external.period_averages import (
    PeriodAverageAccumulator,
    PeriodAverageSample,
)


def test_period_average_accumulator_preserves_nanmean_counts() -> None:
    accumulator = PeriodAverageAccumulator()

    accumulator.add_samples(
        {"temp": PeriodAverageSample(("x",), np.asarray([1.0, np.nan, np.nan]))}
    )
    accumulator.add_samples(
        {"temp": PeriodAverageSample(("x",), np.asarray([3.0, 5.0, np.nan]))}
    )

    accumulated = accumulator.variables["temp"]
    mean_sample = accumulator.mean_samples()["temp"]

    assert mean_sample.dims == ("x",)
    assert isinstance(accumulated.sum_values, jax.Array)
    assert isinstance(accumulated.counts, jax.Array)
    assert isinstance(mean_sample.values, jax.Array)
    assert accumulated.counts.dtype == jnp.int32
    assert_allclose_compact(accumulated.counts, np.asarray([2, 1, 0]))
    assert_allclose_compact(mean_sample.values, np.asarray([2.0, 5.0, np.nan]))


def test_period_average_accumulator_reduces_named_summation_dimension() -> None:
    accumulator = PeriodAverageAccumulator()

    accumulator.add_samples(
        {
            "temp": PeriodAverageSample(
                ("time", "x"),
                np.asarray([[1.0, np.nan], [3.0, 5.0]]),
            )
        },
        summation_dim="time",
    )

    mean_sample = accumulator.mean_samples()["temp"]

    assert mean_sample.dims == ("x",)
    assert_allclose_compact(
        accumulator.variables["temp"].counts,
        np.asarray([2, 1]),
    )
    assert_allclose_compact(mean_sample.values, np.asarray([2.0, 5.0]))


def test_period_average_accumulator_rejects_changed_variables() -> None:
    accumulator = PeriodAverageAccumulator()
    accumulator.add_samples(
        {"temp": PeriodAverageSample(("x",), np.asarray([1.0, 2.0]))}
    )

    with pytest.raises(ValueError, match="variables changed"):
        accumulator.add_samples(
            {"salt": PeriodAverageSample(("x",), np.asarray([1.0, 2.0]))}
        )


def test_period_average_accumulator_rejects_changed_dimensions_or_shape() -> None:
    accumulator = PeriodAverageAccumulator()
    accumulator.add_samples(
        {"temp": PeriodAverageSample(("x",), np.asarray([1.0, 2.0]))}
    )

    with pytest.raises(ValueError, match="dimensions changed"):
        accumulator.add_samples(
            {"temp": PeriodAverageSample(("y",), np.asarray([1.0, 2.0]))}
        )
    with pytest.raises(ValueError, match="shape changed"):
        accumulator.add_samples(
            {"temp": PeriodAverageSample(("x",), np.asarray([1.0, 2.0, 3.0]))}
        )


def test_period_average_accumulator_reports_empty_and_clears() -> None:
    accumulator = PeriodAverageAccumulator()

    with pytest.raises(ValueError, match="requires at least one sample"):
        accumulator.mean_samples()

    accumulator.add_samples(
        {"temp": PeriodAverageSample(("x",), np.asarray([1.0, 2.0]))}
    )
    assert not accumulator.empty

    accumulator.clear()

    assert accumulator.empty
