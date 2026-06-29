"""Lightweight output adapters for external component period files."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from vercor.calendar import ModelDateTime
from vercor.jax_logging import LoggerLike
from vercor.output.period_averages import (
    AccumulatedPeriodVariable,
    PeriodAverageAccumulator,
    accumulate_output_variables,
    period_mean_output_variables,
)
from vercor.output.period_files import write_period_average_netcdf
from vercor.output.time import TIME_NAME, should_write_period_output
from vercor.output.variables import OutputVariable


class ComponentOutputAdapter:
    """Own period-average output state and writes for one external component."""

    def __init__(
        self,
        *,
        empty_error_message: str,
        time_dim: str = TIME_NAME,
        value_dims_for_sample: Callable[[OutputVariable], Sequence[str]] | None = None,
        dimension_order: Sequence[str] | None = None,
    ) -> None:
        """Create an adapter with component-specific mean-output settings."""

        self._accumulator = PeriodAverageAccumulator()
        self._empty_error_message = empty_error_message
        self._time_dim = time_dim
        self._value_dims_for_sample = value_dims_for_sample
        self._dimension_order = (
            tuple(dimension_order) if dimension_order is not None else None
        )

    @property
    def accumulator(self) -> PeriodAverageAccumulator:
        """Return the owned period-average accumulator."""

        return self._accumulator

    @property
    def empty(self) -> bool:
        """Return whether no period samples are currently accumulated."""

        return self._accumulator.empty

    @property
    def variables(self) -> Mapping[str, AccumulatedPeriodVariable]:
        """Return a read-only view of accumulated variables."""

        return self._accumulator.variables

    def reset(self) -> None:
        """Replace accumulated output state with a fresh empty accumulator."""

        self._accumulator = PeriodAverageAccumulator()

    def accumulate(
        self,
        variables: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None = None,
    ) -> None:
        """Accumulate one component output snapshot or prediction block."""

        accumulate_output_variables(
            self._accumulator,
            variables,
            summation_dim=summation_dim,
        )

    def write_period_average(
        self,
        output: str,
        *,
        build_coordinate_variables: Callable[
            [Mapping[str, OutputVariable]],
            Mapping[str, OutputVariable],
        ],
        build_data_variables: (
            Callable[
                [Mapping[str, OutputVariable]],
                Mapping[str, OutputVariable],
            ]
            | None
        ) = None,
        logger: LoggerLike | None = None,
    ) -> None:
        """Write accumulated mean variables to a period-average NetCDF file."""

        write_period_average_netcdf(
            self._accumulator,
            output,
            build_mean_variables=self._mean_variables,
            build_coordinate_variables=build_coordinate_variables,
            build_data_variables=build_data_variables,
            logger=logger,
        )

    def write_period_average_if_due(
        self,
        *,
        time: datetime | ModelDateTime,
        dt: timedelta,
        output_frequency: str | None,
        output: str | Callable[[Any], str],
        build_coordinate_variables: Callable[
            [Mapping[str, OutputVariable]],
            Mapping[str, OutputVariable],
        ],
        build_data_variables: (
            Callable[
                [Mapping[str, OutputVariable]],
                Mapping[str, OutputVariable],
            ]
            | None
        ) = None,
        logger: LoggerLike | None = None,
    ) -> bool:
        """Write a period file when the configured cadence is reached."""

        if not should_write_period_output(
            time=time,
            dt=dt,
            output_frequency=output_frequency,
        ):
            return False

        output_path = output(time) if callable(output) else output
        self.write_period_average(
            output_path,
            build_coordinate_variables=build_coordinate_variables,
            build_data_variables=build_data_variables,
            logger=logger,
        )
        return True

    def _mean_variables(
        self,
        accumulator: PeriodAverageAccumulator,
    ) -> dict[str, OutputVariable]:
        return period_mean_output_variables(
            accumulator,
            empty_error_message=self._empty_error_message,
            time_dim=self._time_dim,
            value_dims_for_sample=self._value_dims_for_sample,
            dimension_order=self._dimension_order,
        )


__all__ = ["ComponentOutputAdapter"]
