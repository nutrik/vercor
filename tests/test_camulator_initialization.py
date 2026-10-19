"""Native CAMulator input bootstrap, vapor capability and exact-time diagnostics."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import xarray as xr

from tests.assertions import assert_allclose_compact, assert_finite_jvp_vjp
from vercor.components import SetupContext
from vercor.exceptions import ComponentError
from vercor.grids import RectilinearGrid
from vercor.jax_logging import get_default_logger
from vercor.physics import PhysicalConstants
from vercor.setups import CAMulatorConfig
from vercor.setups._external import camulator_fields as fields_module
from vercor.setups._external import camulator_gcm_state as state_module
from vercor.setups._external.camulator_tensors import StateVariableAccessor

pytestmark = pytest.mark.fast_always


def test_camulator_config_owns_initial_diagnostics() -> None:
    source = {"time": datetime(2000, 1, 1), "TS": np.full((2, 3), 290.0)}
    config = CAMulatorConfig(
        config_path="native.yml",
        initial_diagnostics=source,
        water_vapor_variable="VAPOR",
    )
    source["TS"][0, 0] = 999.0  # type: ignore[index]
    source["time"] = datetime(2001, 1, 1)
    assert config.initial_diagnostics is not None
    assert config.initial_diagnostics["time"] == datetime(2000, 1, 1)
    assert np.asarray(config.initial_diagnostics["TS"])[0, 0] == 290.0
    with pytest.raises(TypeError):
        cast(Any, config.initial_diagnostics)["TS"] = 0


@pytest.fixture
def native_case(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, SetupContext, dict[str, Any]]:
    start = datetime(2000, 1, 1)
    conf = {
        "data": {
            "variables": ["U", "V", "T", "VAPOR"],
            "surface_variables": ["PS"],
            "diagnostic_variables": ["TS", "FSNS", "FLNS"],
            "dynamic_forcing_variables": ["SST"],
            "forcing_variables": [],
            "static_variables": [],
            "static_first": False,
            "save_loc_physics": "physics.nc",
        },
        "model": {"levels": 2},
        "predict": {"start_datetime": start},
    }
    # Latest frame has distinct physical meteorology after input inverse scaling.
    model_state = torch.zeros((1, 10, 2, 2, 3), dtype=torch.float64)
    model_state[:, :, 0] = -100
    model_state[:, 0:2, -1] = 2
    model_state[:, 2:4, -1] = 1
    model_state[:, 4:6, -1] = 140
    model_state[:, 6:8, -1] = 0.002
    model_state[:, 8:9, -1] = 50000
    calls: dict[str, Any] = {"input": 0, "output": 0, "model": 0, "trace": 0}

    class Transformer:
        def inverse_transform_input(self, value: torch.Tensor) -> torch.Tensor:
            calls["input"] += 1
            return value * 2

        def inverse_transform(self, value: torch.Tensor) -> torch.Tensor:
            calls["output"] += 1
            raise AssertionError("setup must decode input, never prediction")

    def model(value: torch.Tensor) -> torch.Tensor:
        calls["model"] += 1
        return value

    resources = cast(
        Any,
        state_module.CAMulatorGCMSetupState.__new__(
            state_module.CAMulatorGCMSetupState
        ),
    )
    resources.name = "ATM"
    resources.initial_model_state = model_state
    resources.init_noise = None
    resources.stepper = SimpleNamespace(model=model)
    resources.lead_time_periods = 6
    resources.conf = conf
    resources.state_transformer = Transformer()
    resources.forcing_ds_norm = xr.Dataset(
        {"SST": (("time", "lat", "lon"), np.ones((1, 2, 3)))}, coords={"time": [start]}
    )
    resources.df_vars = ["SST"]
    resources.device = "cpu"
    resources.time_alignment = "strict"
    resources.water_vapor_variable = "VAPOR"
    resources.initial_diagnostics = {
        "time": start,
        "TS": np.full((2, 3), 290.0),
        "FSNS": np.full((2, 3), 200.0 * 21600),
        "FLNS": np.full((2, 3), -50.0 * 21600),
    }
    resources.grid = RectilinearGrid(
        name="ATM", longitude=jnp.arange(3.0), latitude=jnp.arange(2.0)
    )
    physics = xr.Dataset(
        {
            "hyai": ("interface", [0.0, 0.0, 0.0]),
            "hybi": ("interface", [0.1, 0.8, 1.0]),
            "hyam": ("level", [0.0, 0.0]),
            "hybm": ("level", [0.45, 0.9]),
            "LANDM_COSLAT": (("lat", "lon"), np.zeros((2, 3))),
        }
    )
    monkeypatch.setattr(state_module.xr, "open_dataset", lambda path: physics)

    def trace(model: Any, dummy: Any) -> Any:
        assert calls["input"] == 1
        calls["trace"] += 1
        return model

    monkeypatch.setattr(state_module.torch.jit, "trace", trace)
    return (
        resources,
        SetupContext(
            start=start, dt_seconds=21600.0, run_order=(), logger=get_default_logger()
        ),
        calls,
    )


def test_native_bootstrap_maps_latest_input_without_advancing(native_case: Any) -> None:
    resources, context, calls = native_case
    original = resources.initial_model_state.clone()
    result = resources.setup(cast(Any, None), context)
    assert calls == {"input": 1, "output": 0, "model": 0, "trace": 1}
    assert torch.equal(original, resources.initial_model_state)
    assert result.payload.model_state is resources.initial_model_state
    assert result.payload.cursor.timestep_counter == 0
    assert result.payload.cursor.start_ix == 0
    assert result.payload.cursor.init_datetime == context.start
    assert result.payload.cursor.model_substeps == 1
    assert result.payload.forecast_hour == 1
    assert result.payload.output_prediction is None
    assert result.payload.output_prediction_samples is None
    mapped = result.fields
    assert all(isinstance(value, jax.Array) for value in mapped.values())
    for key, expected in {
        "temperature": 280.0,
        "specific_humidity": 0.004,
        "u_velocity": 4.0,
        "v_velocity": 2.0,
        "surface_pressure": 100000.0,
        "pressure": 90000.0,
        "net_shortwave_radiation_flux": 200.0,
        "total_surface_temperature": 290.0,
    }.items():
        assert_allclose_compact(mapped[key], np.full((2, 3), expected))
    constants = context.constants
    assert_allclose_compact(
        mapped["potential_temperature"],
        280 * (100000 / 90000) ** constants.dry_air_kappa,
    )
    assert_allclose_compact(
        mapped["density"],
        constants.dry_air_molecular_weight
        / constants.universal_gas_constant
        * 90000
        / 280,
    )
    # One bottom layer with ps=100000, top interface=.8ps: exact IFS alpha.
    alpha = 1 - 0.8 / 0.2 * np.log(1 / 0.8)
    geopotential = (
        constants.dry_air_gas_constant
        * 280
        * (1 + constants.water_vapor_mass_ratio_correction * 0.004)
        * alpha
    )
    geometric = (
        constants.earth_radius
        * geopotential
        / (constants.gravity * constants.earth_radius - geopotential)
    )
    assert_allclose_compact(mapped["model_level_height"], geometric)
    assert_allclose_compact(
        mapped["downward_longwave_radiation_flux"],
        constants.stefan_boltzmann_constant * 290**4 - 50,
    )
    assert mapped["temperature_3d"].shape == (2, 2, 3)
    assert mapped["specific_humidity_3d"].shape == (2, 2, 3)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_time",
        "mismatch_time",
        "missing_TS",
        "missing_FSNS",
        "missing_FLNS",
        "shape",
        "nonfinite",
        "missing_vapor",
        "total_water",
        "missing_input",
        "missing_output",
    ],
)
def test_bootstrap_rejects_missing_physical_capability_before_native_work(
    native_case: Any, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources, context, calls = native_case
    expected = "CAMulator"
    if problem == "missing_time":
        resources.initial_diagnostics.pop("time")
    elif problem == "mismatch_time":
        resources.initial_diagnostics["time"] = context.start + timedelta(hours=6)
    elif problem.startswith("missing_") and problem[8:] in {"TS", "FSNS", "FLNS"}:
        resources.initial_diagnostics.pop(problem[8:])
    elif problem == "shape":
        resources.initial_diagnostics["TS"] = np.zeros((3, 2))
    elif problem == "nonfinite":
        resources.initial_diagnostics["FSNS"][0, 0] = np.nan
    elif problem == "missing_vapor":
        resources.water_vapor_variable = None
    elif problem == "total_water":
        resources.water_vapor_variable = "Qtot"
        resources.conf["data"]["variables"][-1] = "Qtot"
        expected = "total water"
    elif problem == "missing_input":
        resources.conf["data"]["variables"].remove("VAPOR")
        resources.conf["data"]["diagnostic_variables"].append("VAPOR")
    elif problem == "missing_output":
        from vercor.setups._external import camulator_tensors

        def accessor(conf: Any, tensor_type: Any) -> StateVariableAccessor:
            result = StateVariableAccessor(conf, tensor_type)
            if tensor_type == "output":
                result.var_indices["output"]["VAPOR"] = (
                    camulator_tensors.TensorVariableIndex.unavailable(
                        reason="Missing prediction vapor channel"
                    )
                )
            return result

        monkeypatch.setattr(camulator_tensors, "StateVariableAccessor", accessor)
        expected = "CAMulator.*output.*Missing prediction vapor channel"
    with pytest.raises(ComponentError, match=expected):
        resources.setup(cast(Any, None), context)
    assert calls == {"input": 0, "output": 0, "model": 0, "trace": 0}


def test_forcing_start_diagnostics_use_effective_cursor_time(native_case: Any) -> None:
    resources, context, _ = native_case
    resources.time_alignment = "forcing_start"
    context = SetupContext(
        start=context.start + timedelta(days=1),
        dt_seconds=context.dt_seconds,
        run_order=(),
        logger=context.logger,
    )
    result = resources.setup(cast(Any, None), context)
    assert result.payload.cursor.init_datetime == resources.initial_diagnostics["time"]
    assert_allclose_compact(result.fields["temperature"], 280.0)


def test_bootstrap_exports_drive_physical_bulk_forcing(native_case: Any) -> None:
    from vercor.fluxes.bulk_formula_cesm import compute_ocean_surface_fluxes

    resources, context, _ = native_case
    mapped = resources.setup(cast(Any, None), context).fields
    fluxes = compute_ocean_surface_fluxes(
        context.constants,
        jnp.ones((2, 3)),
        mapped["model_level_height"],
        mapped["u_velocity"],
        mapped["v_velocity"],
        mapped["potential_temperature"],
        mapped["specific_humidity"],
        mapped["density"],
        mapped["temperature"],
        jnp.zeros((2, 3)),
        jnp.zeros((2, 3)),
        jnp.full((2, 3), 290.0),
    )
    assert all(np.all(np.isfinite(np.asarray(value))) for value in fluxes)
    # CESM stresses follow wind; downward turbulent heat cools warm ocean.
    assert np.all(np.asarray(fluxes[4]) > 0)
    assert np.all(np.asarray(fluxes[5]) > 0)
    assert np.all(np.asarray(fluxes[0]) < 0)
    assert np.all(np.asarray(fluxes[1]) < 0)


@pytest.mark.parametrize("variable", ["T", "PS", "VAPOR", "U"])
def test_initial_active_nonfinite_meteorology_is_component_scoped(
    native_case: Any, variable: str
) -> None:
    resources, context, calls = native_case
    accessor = StateVariableAccessor(resources.conf, "input")
    index = accessor.get_var_index(variable)
    resources.initial_model_state[0, index.start_idx, -1, 0, 0] = float("nan")
    with pytest.raises(ComponentError, match="CAMulator.*non-finite"):
        resources.setup(cast(Any, None), context)
    assert calls["model"] == 0


def test_shared_meteorology_traces_constants_and_native_fields(
    native_case: Any,
) -> None:
    resources, context, _ = native_case
    resources.setup(cast(Any, None), context)

    def mapped(delta: jax.Array) -> jax.Array:
        from dataclasses import replace

        constants = replace(
            context.constants,
            gravity=context.constants.gravity + delta[0],
            dry_air_gas_constant=context.constants.dry_air_gas_constant + delta[1],
        )
        result = fields_module._map_camulator_air_fields(
            constants,
            reference_pressure=100000.0,
            hyai=jnp.asarray([0.0, 0.0, 0.0]),
            hybi=jnp.asarray([0.1, 0.8, 1.0]),
            hyam=jnp.zeros(2),
            hybm=jnp.asarray([0.45, 0.9]),
            u_wind=jnp.full((2, 2, 3), 4.0) + delta[2],
            v_wind=jnp.full((2, 2, 3), 2.0) + delta[3],
            temperature_3d=jnp.full((2, 2, 3), 280.0) + delta[4],
            water_vapor_3d=jnp.full((2, 2, 3), 0.004) + delta[5],
            surface_pressure=jnp.full((2, 3), 100000.0) + delta[6],
        )
        return cast(jax.Array, sum(jnp.mean(value) for value in result.values()))

    point = jnp.zeros(7)
    assert_finite_jvp_vjp(mapped, point, jnp.ones(7), rtol=1e-7, atol=1e-8)
    gradient = jax.grad(mapped)(point)
    for index, step in enumerate([1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-6, 0.1]):
        delta = jnp.eye(7)[index] * step
        finite_difference = (mapped(delta) - mapped(-delta)) / (2 * step)
        assert_allclose_compact(
            gradient[index], finite_difference, rtol=1e-5, atol=2e-6
        )


def test_prediction_wrapper_requires_vapor_before_inverse(native_case: Any) -> None:
    resources, _, calls = native_case
    with pytest.raises(ComponentError, match="total water"):
        fields_module.map_camulator_prediction_to_runtime_fields(
            PhysicalConstants(),
            camulator_reference_pressure=100000.0,
            native_step_seconds=21600.0,
            water_vapor_variable="Qtot",
            hyai=cast(Any, None),
            hybi=cast(Any, None),
            hyam=cast(Any, None),
            hybm=cast(Any, None),
            accessor_output=StateVariableAccessor(resources.conf, "output"),
            state_transformer=resources.state_transformer,
            prediction=resources.initial_model_state,
        )
    assert calls["output"] == 0


def test_prediction_and_initial_mappers_share_selected_vapor(native_case: Any) -> None:
    resources, context, calls = native_case
    initial = resources.setup(cast(Any, None), context).fields
    accessor = StateVariableAccessor(resources.conf, "output")
    prediction = torch.zeros((1, 12, 1, 2, 3), dtype=torch.float64)
    for name in ("U", "V", "T", "VAPOR", "PS"):
        accessor.set_state_var(
            prediction,
            name,
            resources.accessor_input.get_state_var(
                resources.initial_model_state, name, time_idx=-1
            ).unsqueeze(2),
        )
    for name in ("TS", "FSNS", "FLNS"):
        accessor.set_state_var(
            prediction,
            name,
            torch.as_tensor(resources.initial_diagnostics[name]).reshape(1, 1, 1, 2, 3)
            / 2,
        )

    def inverse(value: torch.Tensor) -> torch.Tensor:
        calls["output"] += 1
        return value * 2

    resources.state_transformer.inverse_transform = inverse
    mapped = fields_module.map_camulator_prediction_to_runtime_fields(
        context.constants,
        camulator_reference_pressure=resources.P0,
        native_step_seconds=resources.model_timestep.total_seconds(),
        water_vapor_variable="VAPOR",
        hyai=resources.hyai,
        hybi=resources.hybi,
        hyam=resources.hyam,
        hybm=resources.hybm,
        accessor_output=accessor,
        state_transformer=resources.state_transformer,
        prediction=prediction,
    )
    for name, value in mapped.items():
        assert_allclose_compact(value, initial[name])
    assert calls["input"] == calls["output"] == 1


def test_initial_noise_is_sampled_once_and_payload_owns_same_sample(
    native_case: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources, context, _ = native_case
    from vercor.setups._external import camulator_init

    samples: list[torch.Tensor] = []

    def sample(value: torch.Tensor, *, noise_std: float, logger: Any) -> torch.Tensor:
        assert noise_std == 0.01
        result = value + 0.01
        samples.append(result)
        return result

    monkeypatch.setattr(camulator_init, "add_init_noise", sample)
    resources.init_noise = 0.01
    result = resources.setup(cast(Any, None), context)
    assert len(samples) == 1
    assert result.payload.model_state is samples[0]
    assert result.payload.model_state is not resources.initial_model_state
    assert_allclose_compact(result.fields["temperature"], 280.02)
    assert result.payload.cursor.timestep_counter == 0


def test_initial_mapper_does_not_call_stepper_or_modify_cursor(
    native_case: Any,
) -> None:
    resources, context, calls = native_case
    result = resources.setup(cast(Any, None), context)
    cursor = result.payload.cursor
    resources.stepper = None
    mapped = fields_module.map_camulator_initial_runtime_fields(
        resources, resources.initial_model_state, cursor, context.constants
    )
    assert_allclose_compact(mapped["temperature"], result.fields["temperature"])
    assert result.payload.cursor is cursor
    assert cursor.timestep_counter == 0
    assert calls["model"] == 0
    assert calls["trace"] == 1


@pytest.mark.parametrize("channel", [None, "Qtot", "ABSENT"])
def test_gallery_rejects_unavailable_vapor_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, channel: str | None
) -> None:
    import io
    from vercor.setups.gallery import run_camulator_with_veros as gallery

    if channel is None:
        monkeypatch.delenv("VERCOR_CAMULATOR_WATER_VAPOR_VARIABLE", raising=False)
    else:
        monkeypatch.setenv("VERCOR_CAMULATOR_WATER_VAPOR_VARIABLE", channel)
    monkeypatch.setattr(
        gallery,
        "open",
        lambda path: io.StringIO("data:\n  variables: [U, V, T, Qtot]\n"),
        raising=False,
    )

    def no_model(**kwargs: Any) -> Any:
        raise AssertionError(
            "invalid gallery capability must fail before any model construction"
        )

    monkeypatch.setattr(gallery, "make_veros_gcm", no_model)
    monkeypatch.setattr(gallery, "make_camulator_gcm", no_model)
    with pytest.raises(ValueError, match="CAMulator gallery requires"):
        gallery.run_setup(loglevel="info", float_type="float64")


def test_gallery_loads_exact_native_initial_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import io
    from vercor.setups.gallery import run_camulator_with_veros as gallery

    start = datetime(2000, 1, 1)
    path = tmp_path / "native-initial.nc"
    xr.Dataset(
        {
            name: (
                ("time", "latitude", "longitude"),
                np.stack([np.full((2, 3), value), np.full((2, 3), 999.0)]),
            )
            for name, value in (
                ("TS", 290.0),
                ("FSNS", 21600 * 200.0),
                ("FLNS", -21600 * 50.0),
            )
        },
        coords={"time": [start, start + timedelta(hours=6)]},
    ).to_netcdf(path, engine="h5netcdf")
    monkeypatch.setenv("VERCOR_CAMULATOR_WATER_VAPOR_VARIABLE", "VAPOR")
    monkeypatch.setenv("VERCOR_CAMULATOR_INITIAL_DIAGNOSTICS", str(path))
    monkeypatch.setattr(
        gallery,
        "open",
        lambda filename: io.StringIO(
            "data:\n  variables: [U, V, T, VAPOR]\npredict:\n  start_datetime: '2000-01-01 00:00:00'\n"
        ),
        raising=False,
    )
    monkeypatch.setattr(gallery, "make_veros_gcm", lambda **kwargs: None)
    captured: list[CAMulatorConfig] = []

    def capture(*, config: CAMulatorConfig) -> Any:
        captured.append(config)
        raise RuntimeError("captured native configuration")

    monkeypatch.setattr(gallery, "make_camulator_gcm", capture)
    with pytest.raises(RuntimeError, match="captured native configuration"):
        gallery.run_setup(loglevel="info", float_type="float64")
    assert captured[0].water_vapor_variable == "VAPOR"
    diagnostics = captured[0].initial_diagnostics
    assert diagnostics is not None
    assert diagnostics["time"] == start
    for name, expected in (
        ("TS", 290.0),
        ("FSNS", 21600 * 200.0),
        ("FLNS", -21600 * 50.0),
    ):
        assert_allclose_compact(diagnostics[name], expected)
    # Exact timestamp selection fails before native construction if absent.
    monkeypatch.setattr(
        gallery,
        "open",
        lambda filename: io.StringIO(
            "data:\n  variables: [VAPOR]\npredict:\n  start_datetime: '2000-01-02 00:00:00'\n"
        ),
        raising=False,
    )
    with pytest.raises(KeyError, match="time"):
        gallery.run_setup(loglevel="info", float_type="float64")
    assert len(captured) == 1


@pytest.mark.parametrize("grid_shape", [(1, 3), (2, 1), (1, 1)])
def test_prediction_mapping_retains_singleton_native_grid_axes(
    grid_shape: tuple[int, int],
) -> None:
    constants = PhysicalConstants()
    air = jnp.full((2, *grid_shape), 280.0)
    surface = jnp.full(grid_shape, 100000.0)
    mapped = fields_module.map_camulator_prediction_arrays(
        constants.earth_radius,
        constants.gravity,
        constants.dry_air_gas_constant,
        constants.water_vapor_mass_ratio_correction,
        constants.dry_air_molecular_weight,
        constants.universal_gas_constant,
        constants.reference_pressure,
        constants.dry_air_kappa,
        constants.stefan_boltzmann_constant,
        100000.0,
        jnp.zeros(3),
        jnp.asarray([0.1, 0.8, 1.0]),
        jnp.zeros(2),
        jnp.asarray([0.45, 0.9]),
        jnp.ones_like(air),
        jnp.ones_like(air),
        jnp.full((1, 1, 1, *grid_shape), 290.0),
        air,
        jnp.full_like(air, 0.004),
        jnp.full((1, 1, 1, *grid_shape), 200 * 21600.0),
        jnp.full((1, 1, 1, *grid_shape), -50 * 21600.0),
        surface.reshape(1, 1, 1, *grid_shape),
        native_step_seconds=21600.0,
    )
    for name, value in mapped.items():
        expected = (2, *grid_shape) if name.endswith("_3d") else grid_shape
        assert value.shape == expected, name


@pytest.mark.parametrize(
    "timestamp",
    [np.datetime64("2000-01-01", "ns"), np.asarray(np.datetime64("2000-01-01", "ns"))],
)
def test_bootstrap_matches_native_numpy_timestamp_exactly(
    native_case: Any, timestamp: Any
) -> None:
    resources, context, _ = native_case
    resources.initial_diagnostics["time"] = timestamp
    result = resources.setup(cast(Any, None), context)
    assert_allclose_compact(result.fields["temperature"], 280.0)
