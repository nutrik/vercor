from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest

from vercor.calendar import DateTime360, DateTime365
from vercor.components import ComponentSpec
from vercor.setups._time_helpers import (
    align_model_timestep,
    assign_model_timestep_alignment,
    runtime_forcing_index,
    run_logged_spinup,
    grid_field_defaults,
)
import vercor.setups._external.camulator_forcing as camulator_forcing_module
from vercor.setups._external.camulator_forcing import (
    initialize_camulator_forcing_cursor,
)
from tests._coverage_support import make_test_grid
from vercor.output import OutputSpec, PeriodOutput
from vercor.setups import JCMAtmosphereConfig, JCMLandAtmosphereConfig, Spinup


class _RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: object, *args: Any, **kwargs: Any) -> None:
        _ = message, args, kwargs

    def info(self, message: object, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self.infos.append(str(message))

    def warning(self, message: object, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self.warnings.append(str(message))

    def error(self, message: object, *args: Any, **kwargs: Any) -> None:
        _ = message, args, kwargs

    def setLevel(self, level: int | str) -> None:
        _ = level

    def isEnabledFor(self, level: int) -> bool:
        _ = level
        return True


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


def test_grid_field_defaults_returns_defaults_with_overrides() -> None:
    fields = grid_field_defaults(
        ("air_temperature", "humidity"),
        overrides={"air_temperature": 280.0},
    )

    assert fields == {"air_temperature": 280.0, "humidity": 0.0}


def test_load_jcm_inputs_facade_returns_named_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vercor.setups as setups
    import vercor.setups._external.jax_gcm_tools as jax_gcm_tools

    coords = object()
    terrain = object()
    forcing = object()
    calls: dict[str, object] = {}

    def fake_load(
        *,
        resolution: int = 31,
        input_data_directory: Path | None = None,
    ) -> tuple[object, object, object]:
        calls["resolution"] = resolution
        calls["input_data_directory"] = input_data_directory
        return coords, terrain, forcing

    monkeypatch.setattr(
        jax_gcm_tools,
        "load_jcm_coords_terrain_forcing",
        fake_load,
    )

    inputs = setups.load_jcm_inputs(
        resolution=42,
        input_data_directory=tmp_path,
    )

    assert isinstance(inputs, setups.JCMInputs)
    assert inputs.coords is coords
    assert inputs.terrain is terrain
    assert inputs.forcing is forcing
    assert calls == {"resolution": 42, "input_data_directory": tmp_path}


def test_make_jcm_land_atmosphere_accepts_preloaded_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups as setups
    import vercor.setups._jcm as helper

    coords = object()
    forcing = object()
    terrain = SimpleNamespace(
        fmask="original-mask", copy=lambda **updates: SimpleNamespace(**updates)
    )
    inputs = setups.JCMInputs(coords=coords, terrain=terrain, forcing=forcing)
    ocean_grid = make_test_grid(name="ocn-grid")
    land_mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    land: Any = SimpleNamespace(
        grid=SimpleNamespace(binary_mask=land_mask),
        spec=SimpleNamespace(initial_fields={"land_fraction": land_mask}),
    )
    atmosphere: Any = object()
    calls: dict[str, Any] = {}

    def unexpected_load_jcm_inputs() -> object:
        pytest.fail("preloaded JCM inputs should avoid loading inputs again")

    def fake_make_jcm_land(
        received_coords: object,
        received_forcing: object,
        received_grid: object,
        *,
        name: str = "LND",
        output: OutputSpec | None = None,
    ) -> Any:
        calls["land_args"] = (
            received_coords,
            received_forcing,
            received_grid,
            name,
            output,
        )
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

    def fake_load_jcm_factories() -> tuple[Any, Any]:
        return fake_make_jcm_land, fake_make_jax_gcm

    monkeypatch.setattr(
        helper,
        "load_jcm_inputs",
        unexpected_load_jcm_inputs,
    )
    monkeypatch.setattr(helper, "_load_jcm_factories", fake_load_jcm_factories)
    monkeypatch.setattr(helper, "transposed_host_array", fake_transposed_host_array)

    result = helper.make_jcm_land_atmosphere(
        ocean_grid,
        inputs=inputs,
        config=JCMLandAtmosphereConfig(
            atmosphere=JCMAtmosphereConfig(spinup=Spinup(enabled=False)),
        ),
    )

    assert result.land is land
    assert result.atmosphere is atmosphere
    assert result.coords is coords
    assert result.terrain is not terrain
    assert result.terrain.fmask == "patched-mask"
    assert result.forcing is forcing
    assert terrain.fmask == "original-mask"
    assert calls["mask"] is land_mask
    assert calls["land_args"] == (
        coords,
        forcing,
        ocean_grid,
        "LND",
        OutputSpec(),
    )
    assert calls["atmosphere_args"] == (
        coords,
        result.terrain,
        {
            "config": JCMAtmosphereConfig(
                forcing_data=forcing,
                spinup=Spinup(enabled=False),
                jitted=True,
            ),
        },
    )


def test_make_jcm_land_atmosphere_replaces_only_missing_forcing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups._jcm as helper

    extension_token = object()
    caller_forcing = object()

    @dataclass(frozen=True)
    class ExtendedJAXGCMConfig(JCMAtmosphereConfig):
        extension: object = extension_token

    atmosphere_config = ExtendedJAXGCMConfig(
        forcing_data=caller_forcing,
        spinup=Spinup(enabled=False),
    )
    inputs = helper.JCMInputs(
        coords=object(),
        terrain=SimpleNamespace(
            fmask=None, copy=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
        forcing=object(),
    )
    ocean_grid = make_test_grid(name="jcm-replace-ocean")
    land = SimpleNamespace(
        grid=SimpleNamespace(binary_mask=jnp.ones(ocean_grid.shape)),
        spec=ComponentSpec(
            initial_fields={"land_fraction": jnp.ones(ocean_grid.shape)},
            outputs=("land_fraction",),
        ),
    )
    captured: dict[str, JCMAtmosphereConfig] = {}

    def fake_make_jcm_land(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        return land

    def fake_make_jax_gcm(
        coords: object,
        terrain: object,
        *,
        config: JCMAtmosphereConfig,
    ) -> object:
        _ = coords, terrain
        captured["config"] = config
        return object()

    monkeypatch.setattr(
        helper,
        "_load_jcm_factories",
        lambda: (fake_make_jcm_land, fake_make_jax_gcm),
    )

    helper.make_jcm_land_atmosphere(
        ocean_grid,
        inputs=inputs,
        config=JCMLandAtmosphereConfig(atmosphere=atmosphere_config),
    )

    replaced = captured["config"]
    assert isinstance(replaced, ExtendedJAXGCMConfig)
    assert replaced.extension is extension_token
    assert replaced.forcing_data is caller_forcing


def test_initialize_camulator_forcing_cursor_requires_explicit_mismatch_policy() -> (
    None
):
    logger = _RecordingLogger()
    forcing_start = datetime(2000, 1, 1, 6)
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(slice(7, 9))})

    with pytest.raises(ValueError, match="forcing start.*coupler start"):
        initialize_camulator_forcing_cursor(
            conf={"predict": {"start_datetime": forcing_start}},
            dynamic_ds=dynamic_ds,
            coupler_start_datetime=datetime(2000, 1, 1),
            logger=logger,
            time_alignment="strict",
        )

    cursor = initialize_camulator_forcing_cursor(
        conf={"predict": {"start_datetime": forcing_start}},
        dynamic_ds=dynamic_ds,
        coupler_start_datetime=datetime(2000, 1, 1),
        logger=logger,
        time_alignment="forcing_start",
    )

    assert cursor.start_ix == 7
    assert cursor.init_str == "2000-01-01T06Z"
    assert cursor.init_datetime == forcing_start
    assert logger.infos == [
        "Starting integration at time index: 7",
        "Starting integration at time index: 7",
    ]
    assert logger.warnings == []


def test_initialize_camulator_forcing_cursor_accepts_integer_index() -> None:
    logger = _RecordingLogger()
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(3)})

    cursor = initialize_camulator_forcing_cursor(
        conf={"predict": {"start_datetime": "2000-01-01 00:00:00"}},
        dynamic_ds=dynamic_ds,
        coupler_start_datetime=datetime(2000, 1, 1),
        logger=logger,
        time_alignment="strict",
    )

    assert cursor.start_ix == 3
    assert cursor.init_str == "2000-01-01T00Z"
    assert logger.warnings == []


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "coupler_start",
    (
        DateTime360(2000, 1, 1, 0, 0, 0, 0),
        DateTime365(2000, 1, 1, 0, 0, 0, 0),
    ),
)
@pytest.mark.parametrize("time_alignment", ("strict", "forcing_start"))
def test_initialize_camulator_forcing_cursor_rejects_model_calendar_clocks(
    coupler_start: DateTime360 | DateTime365,
    time_alignment: str,
) -> None:
    logger = _RecordingLogger()
    forcing_start = datetime(2000, 1, 1)
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(0)})

    with pytest.raises(ValueError, match="standard-library datetime"):
        initialize_camulator_forcing_cursor(
            conf={"predict": {"start_datetime": forcing_start}},
            dynamic_ds=dynamic_ds,
            coupler_start_datetime=coupler_start,
            logger=logger,
            time_alignment=time_alignment,  # type: ignore[arg-type]
        )

    assert logger.infos == []


def test_camulator_runtime_cursor_initializes_indexes_and_advances() -> None:
    logger = _RecordingLogger()
    forcing_start = datetime(2000, 1, 1)
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(4)})
    cursor = camulator_forcing_module.CamulatorRuntimeCursor.initialize(
        conf={"predict": {"start_datetime": forcing_start}},
        dynamic_ds=dynamic_ds,
        coupler_start_datetime=forcing_start,
        model_substeps=3,
        logger=logger,
        time_alignment="strict",
    )

    assert cursor.start_ix == 4
    assert cursor.init_datetime == forcing_start
    assert cursor.init_str == "2000-01-01T00Z"
    assert cursor.timestep_counter == 0
    assert cursor.current_index() == 4

    advanced = cursor.advanced()
    assert advanced.timestep_counter == cursor.timestep_counter + 1
    assert cursor.timestep_counter == 0
    assert advanced.current_index() == 7


def test_make_jcm_land_atmosphere_patches_mask_and_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups._jcm as helper

    coords = object()
    forcing = object()
    terrain = SimpleNamespace(
        fmask="original-mask", copy=lambda **updates: SimpleNamespace(**updates)
    )
    ocean_grid = make_test_grid(name="ocn-grid")
    land_mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    land: Any = SimpleNamespace(
        grid=SimpleNamespace(binary_mask=land_mask),
        spec=SimpleNamespace(initial_fields={"land_fraction": land_mask}),
    )
    atmosphere: Any = object()
    calls: dict[str, Any] = {}

    def fake_load_jcm_inputs() -> Any:
        calls["load_inputs"] = True
        return helper.JCMInputs(coords=coords, terrain=terrain, forcing=forcing)

    def fake_make_jcm_land(
        received_coords: object,
        received_forcing: object,
        received_grid: object,
        *,
        name: str = "LND",
        output: OutputSpec | None = None,
    ) -> Any:
        calls["land_args"] = (
            received_coords,
            received_forcing,
            received_grid,
            name,
            output,
        )
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

    def fake_load_jcm_factories() -> tuple[Any, Any]:
        return fake_make_jcm_land, fake_make_jax_gcm

    monkeypatch.setattr(helper, "load_jcm_inputs", fake_load_jcm_inputs)
    monkeypatch.setattr(helper, "_load_jcm_factories", fake_load_jcm_factories)
    monkeypatch.setattr(helper, "transposed_host_array", fake_transposed_host_array)

    result = helper.make_jcm_land_atmosphere(
        ocean_grid,
        config=JCMLandAtmosphereConfig(
            atmosphere=JCMAtmosphereConfig(
                custom_parameters={"surface_flux.vgust": 5.01},
                spinup=Spinup(enabled=False),
                output=OutputSpec(period=PeriodOutput(frequency="year")),
                jitted=False,
            ),
            land_name="CUSTOM_LND",
        ),
    )

    assert result.land is land
    assert result.atmosphere is atmosphere
    assert result.coords is coords
    assert result.terrain is not terrain
    assert result.terrain.fmask == "patched-mask"
    assert result.forcing is forcing
    assert terrain.fmask == "original-mask"
    assert calls["load_inputs"] is True
    assert calls["mask"] is land_mask
    assert calls["land_args"] == (
        coords,
        forcing,
        ocean_grid,
        "CUSTOM_LND",
        OutputSpec(),
    )
    assert calls["atmosphere_args"] == (
        coords,
        result.terrain,
        {
            "config": JCMAtmosphereConfig(
                custom_parameters={"surface_flux.vgust": 5.01},
                forcing_data=forcing,
                spinup=Spinup(enabled=False),
                output=OutputSpec(period=PeriodOutput(frequency="year")),
                jitted=False,
            ),
        },
    )
