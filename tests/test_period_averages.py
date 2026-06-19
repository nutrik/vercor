from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import vercor.output.period_averages as period_averages_module
from tests.assertions import assert_allclose_compact
from vercor.output.variables import OutputVariable
from vercor.output.period_averages import (
    PeriodAverageAccumulator,
    accumulate_output_variables,
    period_mean_output_variables,
    period_mean_sample_to_output_variable,
)


def test_period_average_public_api_uses_output_variable_directly() -> None:
    removed_names = {
        "PeriodAverageSample",
        "mean_samples_or_raise",
        "samples_from_output_variables",
    }

    assert removed_names.isdisjoint(set(period_averages_module.__all__))
    for name in removed_names:
        assert not hasattr(period_averages_module, name)


def test_period_average_accumulator_preserves_nanmean_counts() -> None:
    accumulator = PeriodAverageAccumulator()

    accumulator.add_samples(
        {"temp": OutputVariable(("x",), np.asarray([1.0, np.nan, np.nan]))}
    )
    accumulator.add_samples(
        {"temp": OutputVariable(("x",), np.asarray([3.0, 5.0, np.nan]))}
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
            "temp": OutputVariable(
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
    accumulator.add_samples({"temp": OutputVariable(("x",), np.asarray([1.0, 2.0]))})

    with pytest.raises(ValueError, match="variables changed"):
        accumulator.add_samples(
            {"salt": OutputVariable(("x",), np.asarray([1.0, 2.0]))}
        )


def test_period_average_accumulator_rejects_changed_dimensions_or_shape() -> None:
    accumulator = PeriodAverageAccumulator()
    accumulator.add_samples({"temp": OutputVariable(("x",), np.asarray([1.0, 2.0]))})

    with pytest.raises(ValueError, match="dimensions changed"):
        accumulator.add_samples(
            {"temp": OutputVariable(("y",), np.asarray([1.0, 2.0]))}
        )
    with pytest.raises(ValueError, match="shape changed"):
        accumulator.add_samples(
            {"temp": OutputVariable(("x",), np.asarray([1.0, 2.0, 3.0]))}
        )


def test_period_average_accumulator_reports_empty_and_clears() -> None:
    accumulator = PeriodAverageAccumulator()

    with pytest.raises(ValueError, match="requires at least one sample"):
        accumulator.mean_samples()

    accumulator.add_samples({"temp": OutputVariable(("x",), np.asarray([1.0, 2.0]))})
    assert not accumulator.empty

    accumulator.clear()

    assert accumulator.empty


def test_accumulate_output_variables_preserves_attrs_and_reduces_dimension() -> None:
    accumulator = PeriodAverageAccumulator()

    accumulate_output_variables(
        accumulator,
        {
            "temp": OutputVariable(
                ("time", "x"),
                np.asarray([[1.0, np.nan], [3.0, 5.0]]),
                {"units": "K"},
            )
        },
        summation_dim="time",
    )

    mean_sample = accumulator.mean_samples()["temp"]

    assert mean_sample.dims == ("x",)
    assert mean_sample.attrs == {"units": "K"}
    assert_allclose_compact(mean_sample.values, np.asarray([2.0, 5.0]))


def test_period_mean_output_variables_uses_adapter_error_message() -> None:
    accumulator = PeriodAverageAccumulator()

    with pytest.raises(ValueError, match="custom adapter message"):
        period_mean_output_variables(
            accumulator,
            empty_error_message="custom adapter message",
        )


def test_period_mean_sample_to_output_variable_orders_explicit_dimensions() -> None:
    values = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
    variable = period_mean_sample_to_output_variable(
        OutputVariable(("lat", "lon", "level"), values, {"units": "K"}),
        time_dim="time",
        dimension_order=("time", "level", "lat", "lon"),
    )

    assert variable.dims == ("time", "level", "lat", "lon")
    assert variable.attrs == {"units": "K"}
    assert_allclose_compact(variable.values[0], np.transpose(values, axes=(2, 0, 1)))


def test_period_mean_sample_to_output_variable_accepts_output_value_dims() -> None:
    values = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
    variable = period_mean_sample_to_output_variable(
        OutputVariable(("xt", "yt", "zt"), values),
        time_dim="time",
        value_dims=("zt", "yt", "xt"),
    )

    assert variable.dims == ("time", "zt", "yt", "xt")
    assert_allclose_compact(variable.values[0], np.transpose(values, axes=(2, 1, 0)))


def test_period_mean_output_variables_applies_jcm_dimension_order() -> None:
    accumulator = PeriodAverageAccumulator()
    values = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
    accumulator.add_samples(
        {
            "temp": OutputVariable(
                ("lat", "lon", "level"),
                values,
                {"units": "K"},
            )
        }
    )

    variables = period_mean_output_variables(
        accumulator,
        empty_error_message="missing samples",
        time_dim="time",
        dimension_order=("time", "level", "lat", "lon"),
    )

    assert variables["temp"].dims == ("time", "level", "lat", "lon")
    assert variables["temp"].attrs == {"units": "K"}
    assert_allclose_compact(
        variables["temp"].values[0],
        np.transpose(values, axes=(2, 0, 1)),
    )


def test_period_mean_output_variables_applies_variable_specific_value_dims() -> None:
    accumulator = PeriodAverageAccumulator()
    temp_values = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
    psi_values = np.arange(2 * 3, dtype=float).reshape((2, 3))
    accumulator.add_samples(
        {
            "temp": OutputVariable(("xt", "yt", "zt"), temp_values),
            "psi": OutputVariable(("xu", "yu"), psi_values),
        }
    )

    variables = period_mean_output_variables(
        accumulator,
        empty_error_message="missing samples",
        time_dim="time",
        value_dims_for_sample=lambda sample: tuple(reversed(sample.dims)),
    )

    assert variables["temp"].dims == ("time", "zt", "yt", "xt")
    assert_allclose_compact(
        variables["temp"].values[0],
        np.transpose(temp_values, axes=(2, 1, 0)),
    )
    assert variables["psi"].dims == ("time", "yu", "xu")
    assert_allclose_compact(
        variables["psi"].values[0],
        np.transpose(psi_values, axes=(1, 0)),
    )
