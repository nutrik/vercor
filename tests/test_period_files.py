from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import logging
from pathlib import Path

import h5netcdf
import numpy as np
import pytest

from tests._coverage_support import capture_logger_output
from tests.assertions import assert_allclose_compact
from vercor.output.datasets import time_coordinate_variable
from vercor.output.period_averages import (
    PeriodAverageAccumulator,
    period_mean_output_variables,
)
from vercor.output.period_files import write_period_average_netcdf
from vercor.output.variables import OutputVariable


def _accumulator_with_temperature() -> PeriodAverageAccumulator:
    accumulator = PeriodAverageAccumulator()
    accumulator.add_samples(
        {
            "temperature": OutputVariable(
                ("x",),
                np.asarray([1.0, 3.0]),
                {"units": "K"},
            )
        }
    )
    return accumulator


def _mean_variables(
    accumulator: PeriodAverageAccumulator,
) -> dict[str, OutputVariable]:
    return period_mean_output_variables(
        accumulator,
        empty_error_message="missing period samples",
        time_dim="time",
    )


def _coordinate_variables(
    variables: Mapping[str, OutputVariable],
) -> dict[str, OutputVariable]:
    _ = variables
    return {
        "time": time_coordinate_variable(datetime(2000, 1, 2)),
        "x": OutputVariable(("x",), np.asarray([10.0, 20.0])),
    }


def test_write_period_average_netcdf_logs_writes_and_clears(
    tmp_path: Path,
) -> None:
    accumulator = _accumulator_with_temperature()
    output = tmp_path / "period-average.nc"
    logger_name = "VerCOR.test.period-files"
    logger = logging.getLogger(logger_name)

    with capture_logger_output(logger_name) as stream:
        write_period_average_netcdf(
            accumulator,
            str(output),
            build_mean_variables=_mean_variables,
            build_coordinate_variables=_coordinate_variables,
            logger=logger,
        )

    with h5netcdf.File(output, "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.dimensions == ("time", "x")
        assert temperature.attrs["units"] == "K"
        assert_allclose_compact(np.asarray(temperature), np.asarray([[1.0, 3.0]]))
        assert_allclose_compact(np.asarray(actual.variables["x"]), [10.0, 20.0])
        assert actual.variables["time"].attrs["calendar"] == "proleptic_gregorian"
    assert accumulator.empty
    assert stream.getvalue().count(f"Writing output file:  {output}") == 1


def test_write_period_average_netcdf_applies_data_variable_builder(
    tmp_path: Path,
) -> None:
    accumulator = _accumulator_with_temperature()
    output = tmp_path / "period-average-with-metadata.nc"

    def build_data_variables(
        variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        return {
            name: OutputVariable(
                variable.dims,
                variable.values,
                {**dict(variable.attrs), "long_name": "Air temperature"},
            )
            for name, variable in variables.items()
        }

    write_period_average_netcdf(
        accumulator,
        str(output),
        build_mean_variables=_mean_variables,
        build_coordinate_variables=_coordinate_variables,
        build_data_variables=build_data_variables,
    )

    with h5netcdf.File(output, "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.attrs["units"] == "K"
        assert temperature.attrs["long_name"] == "Air temperature"


def test_write_period_average_netcdf_keeps_accumulator_when_write_fails(
    tmp_path: Path,
) -> None:
    accumulator = _accumulator_with_temperature()
    output = tmp_path / "conflicting-dimensions.nc"

    def conflicting_data_variables(
        variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        _ = variables
        return {
            "temperature": OutputVariable(
                ("time", "x"),
                np.asarray([[1.0, 2.0, 3.0]]),
                {"units": "K"},
            )
        }

    with pytest.raises(ValueError, match="dimension 'x'.*existing size 2.*new size 3"):
        write_period_average_netcdf(
            accumulator,
            str(output),
            build_mean_variables=_mean_variables,
            build_coordinate_variables=_coordinate_variables,
            build_data_variables=conflicting_data_variables,
        )

    assert not accumulator.empty
