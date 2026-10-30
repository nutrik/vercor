"""Analytic interval radiation checks through the real CAMulator field mapper."""

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

from tests.assertions import assert_allclose_compact
from vercor.components import StepContext
from vercor.output import OutputContext
from vercor.physics import PhysicalConstants
from vercor.setups._external.camulator_forcing import CamulatorRuntimeCursor
from vercor.setups._external.camulator_gcm_state import CAMulatorRuntimePayload
from vercor.setups._external.camulator_output import camulator_output_provider
from vercor.setups._external.camulator_runtime import step_camulator_runtime
from vercor.setups._external.camulator_tensors import StateVariableAccessor


def _native_window_case(
    windows: int, native_seconds: int
) -> tuple[Any, dict[str, jax.Array], StepContext, CAMulatorRuntimePayload]:
    """Replace unavailable CREDIT inference, retaining real conversion/output code."""

    conf = {
        "model": {"levels": 2},
        "data": {
            "variables": ["U", "V", "T", "VAPOR"],
            "surface_variables": ["PS", "TS", "SST"],
            "diagnostic_variables": ["FSNS", "FLNS"],
            "dynamic_forcing_variables": [],
            "forcing_variables": [],
            "static_variables": [],
            "static_first": False,
        },
        "predict": {},
    }
    accessor = StateVariableAccessor(conf, tensor_type="output")

    class NativeStepper:
        def model(self, model_input: torch.Tensor) -> torch.Tensor:
            # U carries the completed-window count, so replay uses only payload.
            window = int(model_input[0, 0, 0, 0, 0])
            prediction = torch.zeros_like(model_input)
            values = {
                "U": window + 1,
                "V": -(window + 1),
                "T": (270.0, 290.0, 310.0)[window],
                "VAPOR": (0.001, 0.003, 0.005)[window],
                "PS": (90000.0, 100000.0, 110000.0)[window],
                "TS": (200.0, 300.0, 400.0)[window],
                "SST": 0.0,
                "FSNS": (100.0, 300.0, 800.0)[window] * native_seconds,
                "FLNS": (-2.0, -6.0, -10.0)[window] * native_seconds,
            }
            for name, value in values.items():
                index = accessor.get_var_index(name)
                prediction[:, index.channel_slice] = value
            return prediction

        def build_input_with_forcing(
            self, state: torch.Tensor, dynamic: torch.Tensor, static: torch.Tensor
        ) -> torch.Tensor:
            return state

        def _apply_postprocessing(
            self, prediction: torch.Tensor, model_input: torch.Tensor
        ) -> torch.Tensor:
            return prediction

        def shift_state_forward(
            self, state: torch.Tensor, prediction: torch.Tensor
        ) -> torch.Tensor:
            return prediction.clone()

    start = datetime(2000, 1, 1)
    native_dt = timedelta(seconds=native_seconds)
    resources = SimpleNamespace(
        conf=conf,
        model_substeps=windows,
        model_timestep=native_dt,
        dynamic_ds=xr.Dataset(
            {"forcing": (("time", "lat", "lon"), np.ones((windows, 2, 2)))},
            coords={"time": [start + i * native_dt for i in range(windows)]},
        ),
        device="cpu",
        stepper=NativeStepper(),
        static_forcing=torch.zeros((1, 1, 1, 2, 2)),
        LANDM_COSLAT=jnp.zeros((2, 2)),
        accessor_input=StateVariableAccessor(conf, tensor_type="input"),
        accessor_output=accessor,
        state_transformer=SimpleNamespace(inverse_transform=lambda value: value),
        water_vapor_variable="VAPOR",
        P0=100000.0,
        hyai=torch.tensor([0.0, 0.05, 0.10]),
        hybi=torch.tensor([0.0, 0.20, 1.0]),
        hyam=torch.tensor([0.015, 0.025]),
        hybm=torch.tensor([0.15, 0.25]),
        metadata={},
        latlons=SimpleNamespace(
            latitude=SimpleNamespace(values=np.asarray([-30.0, 30.0])),
            longitude=SimpleNamespace(values=np.asarray([0.0, 180.0])),
        ),
    )
    fields = {
        "sea_surface_temperature": jnp.asarray([[280.0, 281.0], [282.0, 283.0]]),
        "land_surface_temperature": jnp.zeros((2, 2)),
    }
    context = StepContext(
        time=start,
        dt_seconds=windows * native_seconds,
        constants=PhysicalConstants(stefan_boltzmann_constant=1e-8),
    )
    payload = CAMulatorRuntimePayload(
        model_state=torch.zeros((1, 13, 1, 2, 2)),
        cursor=CamulatorRuntimeCursor(init_datetime=start, model_substeps=windows),
    )
    return resources, fields, context, payload


@pytest.mark.parametrize("native_seconds", [3600, 10800])
@pytest.mark.parametrize(
    "windows, shortwave, longwave, temperature, humidity, pressure",
    [
        (1, 100.0, 14.0, 270.0, 0.001, 90000.0),
        (2, 200.0, 44.5, 290.0, 0.003, 100000.0),
        (3, 400.0, 335.0 / 3.0, 310.0, 0.005, 110000.0),
    ],
)
def test_radiation_integrates_every_native_window_and_meteorology_is_endpoint(
    native_seconds: int,
    windows: int,
    shortwave: float,
    longwave: float,
    temperature: float,
    humidity: float,
    pressure: float,
) -> None:
    # Catches final-window-only rates, coupling-duration normalization, and
    # averaging TS before sigma TS**4. The literal longwave rates are 14,75,246.
    resources, fields, context, payload = _native_window_case(windows, native_seconds)

    result = step_camulator_runtime(resources, fields, context, payload)

    assert_allclose_compact(result.fields["net_shortwave_radiation_flux"], shortwave)
    assert_allclose_compact(result.fields["downward_longwave_radiation_flux"], longwave)
    for name in ("net_shortwave_radiation_flux", "downward_longwave_radiation_flux"):
        assert result.fields[name].dtype == jnp.float32
    for name, expected in {
        "air_temperature": temperature,
        "temperature_3d": temperature,
        "air_specific_humidity": humidity,
        "specific_humidity_3d": humidity,
        "surface_pressure": pressure,
        "air_u_velocity": windows,
        "air_v_velocity": -windows,
    }.items():
        assert_allclose_compact(result.fields[name], expected, rtol=1e-6)
    assert_allclose_compact(
        result.fields["total_surface_temperature"], fields["sea_surface_temperature"]
    )
    assert result.payload.forecast_hour == windows + 1
    assert result.payload.cursor.timestep_counter == 1
    assert result.payload.cursor.current_index() == windows
    assert payload.cursor.timestep_counter == 0
    assert torch.count_nonzero(payload.model_state).item() == 0
    assert torch.equal(result.payload.model_state, result.payload.output_prediction)
    assert_allclose_compact(result.payload.output_prediction[0, 0, 0], windows)


def test_interval_radiation_preserves_native_prediction_output_history() -> None:
    # Catches replacing output history by an interval-mean native prediction.
    resources, fields, context, payload = _native_window_case(3, 3600)
    result = step_camulator_runtime(resources, fields, context, payload)
    assert context.time is not None

    frame = camulator_output_provider(resources).sample(
        OutputContext(
            component=resources,
            state=cast(Any, None),
            payload=result.payload,
            step=0,
            time=context.time,
            dt=timedelta(seconds=context.dt_seconds),
        )
    )

    assert_allclose_compact(
        frame.variables["TS"].values[:, 0, 0], [200.0, 300.0, 400.0]
    )
    assert_allclose_compact(
        frame.variables["FSNS"].values[:, 0, 0], [360000.0, 1080000.0, 2880000.0]
    )
    assert_allclose_compact(result.fields["net_shortwave_radiation_flux"], 400.0)


@pytest.mark.fast_always
def test_longwave_reconstructs_emission_before_temporal_reduction() -> None:
    # sigma=1e-8: (200 K)**4 emits 16 W/m2, (300 K)**4 emits 81 W/m2.
    # Thus mean downward LW=((16-2)+(81-6))/2=44.5, not 35.0625
    # from the mean TS, and not the endpoint 75.
    resources, fields, context, payload = _native_window_case(2, 3600)

    result = step_camulator_runtime(resources, fields, context, payload)

    assert_allclose_compact(result.fields["downward_longwave_radiation_flux"], 44.5)
