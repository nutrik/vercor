from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from setups._time_helpers import (
    align_model_timestep,
    assign_model_timestep_alignment,
    runtime_forcing_index,
    run_logged_spinup,
    seed_grid_field_defaults,
)
from setups.external.camulator_state import initialize_camulator_forcing_cursor
from tests._coverage_support import make_test_grid
from vercor.components.base import data_component
from vercor.settings import VercorSettings


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


def test_assign_model_timestep_alignment_sets_common_state_attributes() -> None:
    state = SimpleNamespace()

    alignment = assign_model_timestep_alignment(
        state,
        86400.0,
        timedelta(hours=6),
    )

    assert alignment.model_substeps == 4
    assert state.coupling_timestep == timedelta(days=1)
    assert state.model_timestep == timedelta(hours=6)
    assert state.model_substeps == 4


def test_runtime_forcing_index_uses_start_counter_and_model_substeps() -> None:
    assert runtime_forcing_index(start_ix=7, timestep_counter=3, model_substeps=4) == 19


def test_run_logged_spinup_logs_each_step_and_returns_callback_result() -> None:
    logger = _RecordingLogger()
    seen_steps: list[int] = []

    def step(step_number: int) -> int:
        seen_steps.append(step_number)
        return step_number * 10

    result = run_logged_spinup(
        steps=3,
        logger=logger,
        intro_message="Running spinup",
        step_message=lambda step, total: f"Step {step} / {total}",
        step=step,
    )

    assert result == 30
    assert seen_steps == [1, 2, 3]
    assert logger.infos == [
        "Running spinup",
        "Step 1 / 3",
        "Step 2 / 3",
        "Step 3 / 3",
    ]


def test_seed_grid_field_defaults_seeds_component_defaults_with_overrides() -> None:
    component = data_component("ATM", make_test_grid())
    context = SimpleNamespace(settings=VercorSettings())

    seed_grid_field_defaults(
        component,
        ("temperature", "humidity"),
        context,
        overrides={"temperature": 280.0},
    )

    assert set(component.data) == {"temperature", "humidity"}
    assert jnp.all(component.data["temperature"] == 280.0)
    assert jnp.all(component.data["humidity"] == 0.0)


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
