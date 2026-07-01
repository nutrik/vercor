from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

import h5netcdf
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from vercor.output.adapters import ComponentOutputAdapter
from vercor.output.datasets import time_coordinate_variable
from vercor.output.variables import OutputVariable


def _coordinate_variables(
    variables: Mapping[str, OutputVariable],
) -> dict[str, OutputVariable]:
    _ = variables
    return {
        "time": time_coordinate_variable(datetime(2000, 1, 2)),
        "x": OutputVariable(("x",), np.asarray([10.0, 20.0])),
    }


def test_component_output_adapter_accumulates_output_variables() -> None:
    adapter = ComponentOutputAdapter(empty_error_message="missing samples")

    adapter.accumulate(
        {
            "temperature": OutputVariable(
                ("time", "x"),
                np.asarray([[1.0, np.nan], [3.0, 5.0]]),
                {"units": "K"},
            )
        },
        summation_dim="time",
    )

    accumulated = adapter.variables["temperature"]
    mean_sample = adapter.accumulator.mean_samples()["temperature"]

    assert accumulated.dims == ("x",)
    assert accumulated.attrs == {"units": "K"}
    assert_allclose_compact(accumulated.counts, np.asarray([2, 1]))
    assert_allclose_compact(mean_sample.values, np.asarray([2.0, 5.0]))


def test_component_output_adapter_writes_when_period_is_due(
    tmp_path: Path,
) -> None:
    adapter = ComponentOutputAdapter(empty_error_message="missing samples")
    adapter.accumulate(
        {
            "temperature": OutputVariable(
                ("x",),
                np.asarray([1.0, 3.0]),
                {"units": "K"},
            )
        }
    )
    output_times: list[datetime] = []

    def output_for_time(output_time: datetime) -> str:
        output_times.append(output_time)
        return str(tmp_path / "period-average.nc")

    written = adapter.write_period_average_if_due(
        time=datetime(2000, 1, 1),
        dt=timedelta(days=1),
        output_frequency="day",
        output=output_for_time,
        build_coordinate_variables=_coordinate_variables,
    )

    assert written
    assert output_times == [datetime(2000, 1, 1)]
    assert adapter.empty
    with h5netcdf.File(tmp_path / "period-average.nc", "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.dimensions == ("time", "x")
        assert temperature.attrs["units"] == "K"
        assert_allclose_compact(np.asarray(temperature), np.asarray([[1.0, 3.0]]))
        assert_allclose_compact(np.asarray(actual.variables["x"]), [10.0, 20.0])


def test_component_output_adapter_records_and_writes_when_period_is_due(
    tmp_path: Path,
) -> None:
    adapter = ComponentOutputAdapter(empty_error_message="missing samples")

    written = adapter.record_period_average_if_due(
        {
            "temperature": OutputVariable(
                ("sample", "x"),
                np.asarray([[1.0, 3.0], [5.0, np.nan]]),
                {"units": "K"},
            )
        },
        summation_dim="sample",
        time=datetime(2000, 1, 1),
        dt=timedelta(days=1),
        output_frequency="day",
        output=str(tmp_path / "period-average.nc"),
        build_coordinate_variables=_coordinate_variables,
    )

    assert written
    assert adapter.empty
    with h5netcdf.File(tmp_path / "period-average.nc", "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.dimensions == ("time", "x")
        assert temperature.attrs["units"] == "K"
        assert_allclose_compact(np.asarray(temperature), np.asarray([[3.0, 3.0]]))


def test_component_output_adapter_skips_when_period_is_not_due(
    tmp_path: Path,
) -> None:
    adapter = ComponentOutputAdapter(empty_error_message="missing samples")
    adapter.accumulate({"temperature": OutputVariable(("x",), np.asarray([1.0, 3.0]))})

    def output_for_time(output_time: datetime) -> str:
        _ = output_time
        raise AssertionError("output path should not be built when period is not due")

    written = adapter.write_period_average_if_due(
        time=datetime(2000, 1, 1),
        dt=timedelta(hours=1),
        output_frequency="day",
        output=output_for_time,
        build_coordinate_variables=_coordinate_variables,
    )

    assert not written
    assert not adapter.empty
    assert not (tmp_path / "period-average.nc").exists()


def test_component_output_adapter_records_and_retains_when_period_is_not_due(
    tmp_path: Path,
) -> None:
    adapter = ComponentOutputAdapter(empty_error_message="missing samples")

    def output_for_time(output_time: datetime) -> str:
        _ = output_time
        raise AssertionError("output path should not be built when period is not due")

    written = adapter.record_period_average_if_due(
        {"temperature": OutputVariable(("x",), np.asarray([1.0, 3.0]))},
        time=datetime(2000, 1, 1),
        dt=timedelta(hours=1),
        output_frequency="day",
        output=output_for_time,
        build_coordinate_variables=_coordinate_variables,
    )

    assert not written
    assert not adapter.empty
    assert not (tmp_path / "period-average.nc").exists()
    assert_allclose_compact(
        adapter.accumulator.mean_samples()["temperature"].values,
        np.asarray([1.0, 3.0]),
    )


def test_component_output_adapter_keeps_samples_when_write_fails(
    tmp_path: Path,
) -> None:
    adapter = ComponentOutputAdapter(empty_error_message="missing samples")
    adapter.accumulate({"temperature": OutputVariable(("x",), np.asarray([1.0, 3.0]))})

    def conflicting_data_variables(
        variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        _ = variables
        return {
            "temperature": OutputVariable(
                ("time", "x"),
                np.asarray([[1.0, 2.0, 3.0]]),
            )
        }

    with pytest.raises(ValueError, match="dimension 'x'.*existing size 2.*new size 3"):
        adapter.write_period_average(
            str(tmp_path / "conflicting-dimensions.nc"),
            build_coordinate_variables=_coordinate_variables,
            build_data_variables=conflicting_data_variables,
        )

    assert not adapter.empty


def test_component_output_adapter_record_keeps_samples_when_write_fails(
    tmp_path: Path,
) -> None:
    adapter = ComponentOutputAdapter(empty_error_message="missing samples")

    def conflicting_data_variables(
        variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        _ = variables
        return {
            "temperature": OutputVariable(
                ("time", "x"),
                np.asarray([[1.0, 2.0, 3.0]]),
            )
        }

    with pytest.raises(ValueError, match="dimension 'x'.*existing size 2.*new size 3"):
        adapter.record_period_average_if_due(
            {"temperature": OutputVariable(("x",), np.asarray([1.0, 3.0]))},
            time=datetime(2000, 1, 1),
            dt=timedelta(days=1),
            output_frequency="day",
            output=str(tmp_path / "conflicting-dimensions.nc"),
            build_coordinate_variables=_coordinate_variables,
            build_data_variables=conflicting_data_variables,
        )

    assert not adapter.empty


def test_component_output_adapter_writes_latest_snapshot_without_clearing_period_state(
    tmp_path: Path,
) -> None:
    adapter = ComponentOutputAdapter(empty_error_message="missing samples")
    adapter.accumulate({"period": OutputVariable(("x",), np.asarray([10.0, 20.0]))})
    adapter.record_snapshot(
        {
            "temperature": OutputVariable(
                ("sample", "x"),
                np.asarray([[1.0, np.nan], [3.0, 5.0]]),
                {"units": "K"},
            )
        },
        summation_dim="sample",
        time=datetime(2000, 1, 2),
    )

    assert not adapter.empty
    assert not adapter.snapshot_empty
    assert adapter.snapshot_time == datetime(2000, 1, 2)
    assert tuple(adapter.snapshot_variables) == ("temperature",)
    assert adapter.snapshot_variables["temperature"].dims == ("x",)

    adapter.write_snapshot(
        str(tmp_path / "snapshot.nc"),
        build_coordinate_variables=_coordinate_variables,
    )

    assert not adapter.empty
    assert not adapter.snapshot_empty
    assert_allclose_compact(
        adapter.accumulator.mean_samples()["period"].values,
        np.asarray([10.0, 20.0]),
    )
    with h5netcdf.File(tmp_path / "snapshot.nc", "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.dimensions == ("time", "x")
        assert temperature.attrs["units"] == "K"
        assert_allclose_compact(
            np.asarray(temperature),
            np.asarray([[2.0, 5.0]]),
        )


def test_component_output_adapter_snapshot_replaces_previous_record_and_resets() -> (
    None
):
    adapter = ComponentOutputAdapter(empty_error_message="missing samples")
    adapter.record_snapshot({"old": OutputVariable(("x",), np.asarray([1.0]))})
    adapter.record_snapshot(
        {"new": OutputVariable(("x",), np.asarray([2.0]))},
        time=datetime(2001, 1, 1),
    )

    assert tuple(adapter.snapshot_variables) == ("new",)
    assert_allclose_compact(
        adapter.snapshot_variables["new"].values,
        np.asarray([2.0]),
    )
    assert adapter.snapshot_time == datetime(2001, 1, 1)

    adapter.accumulate({"period": OutputVariable(("x",), np.asarray([3.0]))})
    adapter.reset()

    assert adapter.empty
    assert adapter.snapshot_empty
    assert adapter.snapshot_time is None
