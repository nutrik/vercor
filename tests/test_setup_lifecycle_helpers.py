from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest

from vercor.setups._time_helpers import (
    align_model_timestep,
    assign_model_timestep_alignment,
    runtime_forcing_index,
    run_logged_spinup,
    seed_grid_field_defaults,
)
import vercor.setups.external.camulator_state as camulator_state_module
import vercor.setups.external.camulator_forcing as camulator_forcing_module
from vercor.setups.external.camulator_forcing import initialize_camulator_forcing_cursor
from tests._coverage_support import make_test_grid
from vercor.components import data_component
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
    with pytest.raises(
        ValueError,
        match=r"model_timestep .* must evenly divide coupling_timestep",
    ):
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


def test_camulator_runtime_cursor_initializes_indexes_and_advances() -> None:
    assert hasattr(camulator_state_module, "CamulatorRuntimeCursor")
    logger = _RecordingLogger()
    forcing_start = datetime(2000, 1, 1)
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(4)})
    cursor = camulator_forcing_module.CamulatorRuntimeCursor()

    cursor.initialize(
        conf={"predict": {"start_datetime": forcing_start}},
        dynamic_ds=dynamic_ds,
        coupler_start_datetime=forcing_start,
        model_substeps=3,
        logger=logger,
    )

    assert cursor.start_ix == 4
    assert cursor.init_datetime == forcing_start
    assert cursor.init_str == "2000-01-01T00Z"
    assert cursor.timestep_counter == 0
    assert cursor.current_index() == 4

    cursor.advance()
    assert cursor.timestep_counter == 1
    assert cursor.current_index() == 7


def test_build_jcm_land_atmosphere_components_patches_mask_and_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups.jcm_setup_helpers as helper

    coords = object()
    forcing = object()
    terrain = SimpleNamespace(fmask="original-mask")
    ocean_grid = make_test_grid(name="ocn-grid")
    land_mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    land: Any = SimpleNamespace(grid=SimpleNamespace(binary_mask=land_mask))
    atmosphere: Any = object()
    calls: dict[str, Any] = {}

    def fake_generate() -> tuple[object, SimpleNamespace, object]:
        calls["generated"] = True
        return coords, terrain, forcing

    def fake_make_jcm_land(
        received_coords: object,
        received_forcing: object,
        received_grid: object,
    ) -> Any:
        calls["land_args"] = (received_coords, received_forcing, received_grid)
        return land

    def fake_transposed_host_array(mask: object) -> str:
        calls["mask"] = mask
        return "patched-mask"

    def fake_make_jax_gcm(
        received_coords: object,
        received_terrain: object,
        **kwargs: object,
    ) -> object:
        calls["atmosphere_args"] = (received_coords, received_terrain, kwargs)
        return atmosphere

    monkeypatch.setattr(
        helper, "generate_jcm_coords_forcing_topography_files", fake_generate
    )
    monkeypatch.setattr(helper, "make_jcm_land", fake_make_jcm_land)
    monkeypatch.setattr(helper, "transposed_host_array", fake_transposed_host_array)
    monkeypatch.setattr(helper, "make_jax_gcm", fake_make_jax_gcm)

    result = helper.build_jcm_land_atmosphere_components(
        ocean_grid,
        custom_parameters={"surface_flux.vgust": 5.01},
        do_spinup=False,
        jitted=False,
        output_frequency="year",
    )

    assert result.land is land
    assert result.atmosphere is atmosphere
    assert result.coords is coords
    assert result.terrain is terrain
    assert result.forcing is forcing
    assert terrain.fmask == "patched-mask"
    assert calls["mask"] is land_mask
    assert calls["land_args"] == (coords, forcing, ocean_grid)
    assert calls["atmosphere_args"] == (
        coords,
        terrain,
        {
            "custom_parameters": {"surface_flux.vgust": 5.01},
            "forcing_data": forcing,
            "do_spinup": False,
            "jitted": False,
            "output_frequency": "year",
        },
    )
