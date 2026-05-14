from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from setups._time_helpers import align_model_timestep
from setups.external.camulator_state import initialize_camulator_forcing_cursor


class _RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _TimeIndex:
    def __init__(self, loc: int | slice) -> None:
        self.loc = loc

    def get_loc(self, value: object) -> int | slice:
        _ = value
        return self.loc


def test_align_model_timestep_returns_coupling_timestep_and_substeps() -> None:
    alignment = align_model_timestep(
        86400.0,
        timedelta(hours=6),
    )

    assert alignment.coupling_timestep == timedelta(days=1)
    assert alignment.model_substeps == 4


def test_align_model_timestep_rejects_non_divisible_model_step() -> None:
    with pytest.raises(ValueError, match="model_timestep"):
        align_model_timestep(
            3600.0,
            timedelta(minutes=45),
        )


def test_initialize_camulator_forcing_cursor_returns_index_and_warns_on_mismatch() -> (
    None
):
    logger = _RecordingLogger()
    forcing_start = datetime(2000, 1, 1, 6)
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(slice(7, 9))})

    cursor = initialize_camulator_forcing_cursor(
        conf={"predict": {"start_datetime": forcing_start}},
        dynamic_ds=dynamic_ds,
        coupler_start_datetime=datetime(2000, 1, 1),
        logger=logger,
    )

    assert cursor.start_ix == 7
    assert cursor.init_str == "2000-01-01T06Z"
    assert cursor.init_datetime == forcing_start
    assert logger.infos == ["Starting integration at time index: 7"]
    assert len(logger.warnings) == 1
    assert "does not match" in logger.warnings[0]


def test_initialize_camulator_forcing_cursor_accepts_integer_index() -> None:
    logger = _RecordingLogger()
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(3)})

    cursor = initialize_camulator_forcing_cursor(
        conf={"predict": {"start_datetime": "2000-01-01 00:00:00"}},
        dynamic_ds=dynamic_ds,
        coupler_start_datetime=datetime(2000, 1, 1),
        logger=logger,
    )

    assert cursor.start_ix == 3
    assert cursor.init_str == "2000-01-01T00Z"
    assert logger.warnings == []
