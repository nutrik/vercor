from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import h5netcdf

from vercor.output.netcdf import write_netcdf_dataset
from vercor.output.variables import OutputVariable


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def _record(self, message: object, *args: Any, **kwargs: Any) -> None:
        _ = kwargs
        message_text = str(message)
        self.messages.append(message_text.format(*args) if args else message_text)

    def debug(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._record(message, *args, **kwargs)

    def info(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._record(message, *args, **kwargs)

    def warning(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._record(message, *args, **kwargs)

    def error(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._record(message, *args, **kwargs)

    def setLevel(self, level: int | str) -> None:
        _ = level

    def isEnabledFor(self, level: int) -> bool:
        _ = level
        return True


def test_write_netcdf_dataset_rejects_conflicting_dimension_sizes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "conflicting-dimensions.nc"

    with pytest.raises(ValueError, match="dimension 'x'.*existing size 2.*new size 3"):
        write_netcdf_dataset(
            output=str(output),
            coordinate_variables={
                "x": OutputVariable(("x",), np.asarray([0.0, 1.0])),
            },
            data_variables={
                "bad": OutputVariable(("x",), np.asarray([0.0, 1.0, 2.0])),
            },
        )


def test_write_netcdf_dataset_logs_filename_when_logger_is_supplied(
    tmp_path: Path,
) -> None:
    output = tmp_path / "logged-output.nc"
    logger = _RecordingLogger()

    write_netcdf_dataset(
        output=str(output),
        coordinate_variables={
            "x": OutputVariable(("x",), np.asarray([0.0, 1.0])),
        },
        data_variables={
            "temperature": OutputVariable(("x",), np.asarray([280.0, 281.0])),
        },
        logger=logger,
    )

    assert logger.messages == [f"Writing output file:  {output}"]


def test_write_netcdf_dataset_writes_scalar_data_variables(
    tmp_path: Path,
) -> None:
    output = tmp_path / "scalar-output.nc"

    write_netcdf_dataset(
        output=str(output),
        coordinate_variables={
            "x": OutputVariable(("x",), np.asarray([0.0, 1.0])),
        },
        data_variables={
            "forecast_hour": OutputVariable((), np.asarray(12, dtype=np.int32)),
        },
    )

    with h5netcdf.File(output, "r") as actual:
        assert int(actual.variables["forecast_hour"][()]) == 12
