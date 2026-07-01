"""Lightweight output adapters for external component period files."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from vercor.calendar import ModelDateTime
from vercor.jax_logging import LoggerLike
from vercor.output.period_averages import (
    AccumulatedPeriodVariable,
    PeriodAverageAccumulator,
    period_mean_output_variables,
)
from vercor.output.period_files import write_period_average_netcdf
from vercor.output.time import TIME_NAME, should_write_period_output
from vercor.output.variables import OutputVariable

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.runtime.state import RuntimeComponentState

ComponentSnapshotWriter: TypeAlias = Callable[
    ["RuntimeComponentState", Path, datetime | ModelDateTime, LoggerLike | None],
    None,
]
_SNAPSHOT_WRITER_METADATA_KEY = "component_snapshot_writer"


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
        self._snapshot_variables: dict[str, OutputVariable] | None = None
        self._snapshot_time: datetime | ModelDateTime | None = None
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

    @property
    def snapshot_empty(self) -> bool:
        """Return whether no single-record snapshot has been recorded."""

        return self._snapshot_variables is None

    @property
    def snapshot_variables(self) -> Mapping[str, OutputVariable]:
        """Return a read-only view of the latest snapshot variables."""

        return MappingProxyType(self._snapshot_variables or {})

    @property
    def snapshot_time(self) -> datetime | ModelDateTime | None:
        """Return the time associated with the latest snapshot record."""

        return self._snapshot_time

    def reset(self) -> None:
        """Replace accumulated output state with a fresh empty accumulator."""

        self._accumulator = PeriodAverageAccumulator()
        self._snapshot_variables = None
        self._snapshot_time = None

    def accumulate(
        self,
        variables: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None = None,
    ) -> None:
        """Accumulate one component output snapshot or prediction block."""

        self._accumulator.add_samples(
            variables,
            summation_dim=summation_dim,
        )

    def record_snapshot(
        self,
        variables: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None = None,
        time: datetime | ModelDateTime | None = None,
    ) -> None:
        """Record one latest component output snapshot without accumulating."""

        snapshot_accumulator = PeriodAverageAccumulator()
        snapshot_accumulator.add_samples(variables, summation_dim=summation_dim)
        self._snapshot_variables = snapshot_accumulator.mean_samples()
        self._snapshot_time = time

    def write_snapshot(
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
        """Write the latest single-record snapshot without clearing it."""

        if self._snapshot_variables is None:
            raise ValueError(self._empty_error_message)

        snapshot_accumulator = PeriodAverageAccumulator()
        snapshot_accumulator.add_samples(self._snapshot_variables)
        write_period_average_netcdf(
            snapshot_accumulator,
            output,
            build_mean_variables=self._mean_variables,
            build_coordinate_variables=build_coordinate_variables,
            build_data_variables=build_data_variables,
            logger=logger,
        )

    def record_period_average_if_due(
        self,
        variables: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None = None,
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
        """Accumulate one sample and write a period file when due."""

        self.accumulate(variables, summation_dim=summation_dim)
        return self.write_period_average_if_due(
            time=time,
            dt=dt,
            output_frequency=output_frequency,
            output=output,
            build_coordinate_variables=build_coordinate_variables,
            build_data_variables=build_data_variables,
            logger=logger,
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


def register_component_snapshot_writer(
    component: "Component",
    writer: ComponentSnapshotWriter,
) -> None:
    """Register a component-owned native snapshot writer for finalization."""

    component.setup_metadata[_SNAPSHOT_WRITER_METADATA_KEY] = writer


def component_snapshot_writer(
    component: "Component",
) -> ComponentSnapshotWriter | None:
    """Return a registered component snapshot writer, if one exists."""

    writer = component.setup_metadata.get(_SNAPSHOT_WRITER_METADATA_KEY)
    if writer is None:
        return None
    if not callable(writer):
        raise TypeError("component snapshot writer metadata must be callable.")
    return cast(ComponentSnapshotWriter, writer)


__all__ = [
    "ComponentOutputAdapter",
    "ComponentSnapshotWriter",
    "component_snapshot_writer",
    "register_component_snapshot_writer",
]
