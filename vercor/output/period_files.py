"""Shared NetCDF write lifecycle for period-average output files."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from vercor.jax_logging import LoggerLike, get_default_logger
from vercor.output.netcdf import write_netcdf_dataset
from vercor.output.period_averages import PeriodAverageAccumulator
from vercor.output.variables import OutputVariable


def write_period_average_netcdf(
    accumulator: PeriodAverageAccumulator,
    output: str,
    *,
    build_mean_variables: Callable[
        [PeriodAverageAccumulator],
        Mapping[str, OutputVariable],
    ],
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
    """Write one period-average NetCDF file and clear accumulated samples."""

    log = logger if logger is not None else get_default_logger()
    log.info(f"Writing output file:  {output:s}")

    mean_variables = build_mean_variables(accumulator)
    data_variables = (
        build_data_variables(mean_variables)
        if build_data_variables is not None
        else mean_variables
    )
    write_netcdf_dataset(
        output=output,
        coordinate_variables=build_coordinate_variables(mean_variables),
        data_variables=data_variables,
    )
    accumulator.clear()


__all__ = ["write_period_average_netcdf"]
