"""Check native-step flux integrals independently of save cadence and endpoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import jax
import jax_datetime as jdt
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact, assert_finite_jvp_vjp
from vercor.dtypes import DTypePolicy
from vercor.physics import PhysicalConstants
from vercor.setups._external.jax_gcm_runtime import (
    JCMAtmosphereRuntimePayload,
    step_jax_gcm_runtime,
)
from vercor.setups._external.jax_gcm_state import JCMAtmosphereSetupState, JCMState

pytestmark = pytest.mark.fast_always


class _Dynamics(NamedTuple):
    u_wind: Any
    v_wind: Any
    temperature: Any
    specific_humidity: Any
    normalized_surface_pressure: Any


class _Surface(NamedTuple):
    shf: Any
    evap: Any
    rlds: Any
    t0: Any


class _Shortwave(NamedTuple):
    rsns: Any
    rsds: Any


class _Condensation(NamedTuple):
    precls: Any


class _Convection(NamedTuple):
    precnv: Any


class _Predictions(NamedTuple):
    dynamics: _Dynamics
    physics: dict[str, Any]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Forcing:
    """Static boundary for isolating real mapper behavior from native dynamics."""

    stl_am: Any
    sea_surface_temperature: Any
    soilw_am: Any
    sice_am: Any

    def copy(self, **updates: Any) -> _Forcing:
        return replace(self, **updates)

    def select(self, date: Any) -> _Forcing:
        _ = date
        return self


def _case(temperatures: Any, *, enable_x64: bool = True) -> tuple[Any, Any, Any]:
    dtype = DTypePolicy(enable_x64=enable_x64)

    def frames(values: Any, *, profile: bool = False) -> jax.Array:
        values = jnp.asarray(values, dtype=dtype.jax_real)
        shape = (2, 2, 2, 2) if profile else (2, 2, 2)
        return jnp.broadcast_to(values.reshape((2,) + (1,) * (len(shape) - 1)), shape)

    predictions = _Predictions(
        _Dynamics(
            frames([1.0, 5.0], profile=True),
            frames([2.0, 6.0], profile=True),
            frames([280.0, 300.0], profile=True),
            frames([0.005, 0.01], profile=True),
            frames([0.9, 1.0]),
        ),
        {
            "_surface_flux": _Surface(
                frames([10.0, 30.0]),
                frames([1.0, 3.0]),
                frames([100.0, 300.0]),
                frames(temperatures),
            ),
            "_shortwave_rad": _Shortwave(frames([0.0, 180.0]), frames([0.0, 200.0])),
            "_condensation": _Condensation(frames([8.0, 28.0])),
            "_convection": _Convection(frames([2.0, 2.0])),
            "category": jnp.asarray([1, 2], dtype=jnp.int32),
        },
    )
    endpoint = jax.tree.map(lambda value: value[-1], predictions)
    native = JCMState(
        dynamics=endpoint.dynamics,
        physics=endpoint.physics,
        dycore_state={"marker": jnp.asarray(3.0, dtype=dtype.jax_real)},
        physics_carry={"marker": jnp.asarray(4.0, dtype=dtype.jax_real)},
        time=jdt.to_datetime("2000-01-01T01:00:00"),
        step=jnp.int32(2),
    )
    forcing = _Forcing(
        *(jnp.full((2, 2), value, dtype=dtype.jax_real) for value in (290, 280, 0.5, 0))
    )
    owner = SimpleNamespace(
        name="ATM",
        seaice_coupling=False,
        prescribed_marine_fraction=None,
        open_water_albedo=jnp.asarray(0.07, dtype=dtype.jax_real),
        sigma_levels=jnp.asarray([0.2, 0.95], dtype=dtype.jax_real),
        model=SimpleNamespace(
            terrain=SimpleNamespace(fmask=jnp.full((2, 2), 0.2, dtype=dtype.jax_real))
        ),
        model_timestep=timedelta(minutes=30),
        save_interval=timedelta(hours=1),
        coupling_timestep=timedelta(hours=1),
        _step_function=lambda state, applied: (
            native,
            {
                "surface_sensible_heat_flux": predictions.physics["_surface_flux"].shf,
                "surface_evaporation": predictions.physics["_surface_flux"].evap,
                "downward_longwave_radiation_flux": predictions.physics[
                    "_surface_flux"
                ].rlds,
                "net_shortwave_radiation_flux": predictions.physics[
                    "_shortwave_rad"
                ].rsns,
                "downward_shortwave_radiation_flux": predictions.physics[
                    "_shortwave_rad"
                ].rsds,
                "large_scale_precipitation": predictions.physics[
                    "_condensation"
                ].precls,
                "convective_precipitation": predictions.physics["_convection"].precnv,
                "near_surface_air_temperature": predictions.physics["_surface_flux"].t0,
            },
        ),
    )
    return owner, JCMAtmosphereRuntimePayload(native, forcing), predictions


def _advance(temperatures: Any, *, enable_x64: bool = True) -> Any:
    owner, payload, _ = _case(temperatures, enable_x64=enable_x64)
    return step_jax_gcm_runtime(
        owner,
        {
            "land_surface_temperature": 290 * jnp.ones((2, 2)),
            "sea_surface_temperature": 280 * jnp.ones((2, 2)),
            "soil_moisture": 0.5 * jnp.ones((2, 2)),
        },
        payload,
        PhysicalConstants(),
        DTypePolicy(enable_x64=enable_x64),
    )


@pytest.mark.parametrize("enable_x64", [False, True])
def test_flux_exports_integrate_both_native_windows(enable_x64: bool) -> None:
    result = _advance([272.77, 278.15], enable_x64=enable_x64)
    for name, expected in {
        "sensible_heat_flux": -20.0,
        "latent_heat_flux": -5002.0,
        "downward_longwave_radiation_flux": 200.0,
        "downward_shortwave_radiation_flux": 100.0,
        "net_shortwave_radiation_flux": 90.0,
    }.items():
        assert_allclose_compact(result.fields[name], expected, rtol=2e-6)
        assert result.fields[name].dtype == DTypePolicy(enable_x64=enable_x64).jax_real
    assert_allclose_compact(result.fields["air_temperature"], 300.0)
    assert_allclose_compact(result.fields["air_u_velocity"], 5.0)
    assert_allclose_compact(result.fields["surface_pressure"], 1e5)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_phase_is_diagnosed_before_temporal_integration(enable_x64: bool) -> None:
    result = _advance([272.77, 278.15], enable_x64=enable_x64)
    # (10*sigmoid(2) + 30*sigmoid(-2))/2, converted g to kg.
    assert_allclose_compact(
        result.fields["snow_rate"], 0.0061920292202211746, rtol=2e-6
    )
    assert_allclose_compact(result.fields["rain_rate"], 0.013807970779778827, rtol=2e-6)
    assert_allclose_compact(
        result.fields["rain_rate"] + result.fields["snow_rate"], 0.02, rtol=2e-6
    )


def test_integrated_phase_supports_jit_and_both_ad_modes() -> None:
    def objective(temperature: jax.Array) -> jax.Array:
        return jnp.sum(_advance(temperature).fields["snow_rate"])

    value = jnp.asarray([272.77, 278.15])
    assert_allclose_compact(jax.jit(objective)(value), 4 * 0.0061920292202211746)
    assert_finite_jvp_vjp(objective, value, jnp.ones_like(value))
    expected = (
        -4 * 4 / 5.38 * np.asarray([0.01, 0.03]) / 2 * np.exp(2) / (1 + np.exp(2)) ** 2
    )
    assert_allclose_compact(jax.grad(objective)(value), expected, rtol=1e-10)


def test_native_sampling_and_endpoint_do_not_follow_save_interval() -> None:
    owner, payload, predictions = _case([272.77, 278.15])
    native = payload.jcm_state
    seen: list[float] = []

    class Model:
        def run_from_state_with_carry(self, **kwargs: Any) -> tuple[Any, Any]:
            seen.append(kwargs["save_interval"])
            return (
                SimpleNamespace(
                    dynamics=native.dycore_state,
                    physics=native.physics_carry,
                    time=native.time,
                    step=native.step,
                ),
                predictions,
            )

    component = JCMAtmosphereSetupState.__new__(JCMAtmosphereSetupState)
    component._dtype_policy = DTypePolicy()
    component.name = "ATM"
    component.model_timestep = timedelta(minutes=30)
    component.coupling_timestep = timedelta(hours=1)
    component.model = cast(Any, Model())
    for save_seconds in (1800, 3600):
        component.save_interval = timedelta(seconds=save_seconds)
        endpoint, samples = component._generate_step_function(jitted=False)(
            native, payload.forcing
        )
        assert_allclose_compact(endpoint.dynamics.u_wind, 5.0)
        assert int(endpoint.physics["category"]) == 2
        assert endpoint.physics["category"].dtype == jnp.int32
        assert int(endpoint.time.delta.seconds) == 3600
        assert int(endpoint.step) == 2
        assert isinstance(samples, dict)
        assert set(samples) == {
            "surface_sensible_heat_flux",
            "surface_evaporation",
            "downward_longwave_radiation_flux",
            "net_shortwave_radiation_flux",
            "downward_shortwave_radiation_flux",
            "large_scale_precipitation",
            "convective_precipitation",
            "near_surface_air_temperature",
        }
    assert_allclose_compact(seen, [1 / 48, 1 / 48])


def test_applied_surface_state_uses_last_native_physics_date() -> None:
    from jcm.forcing import BY_DATE_INTERP, make_time_series
    from tests.test_jcm_surface_forcing import native_case
    from vercor.setups._external.jax_gcm_runtime import _sample_native_surface_forcing

    _, terrain, _, forcing = native_case()
    times = jdt.to_datetime(
        np.asarray(["2000-01-01", "2000-01-02"], dtype="datetime64[s]")
    )
    forcing = forcing.copy(
        sea_surface_temperature=make_time_series(
            jnp.stack([jnp.full(terrain.fmask.shape, t) for t in (270.0, 294.0)]),
            times,
            align_mode=BY_DATE_INTERP,
        )
    )
    result = _sample_native_surface_forcing(
        jdt.to_datetime("2000-01-01"),
        jnp.int32(0),
        forcing,
        model_seconds=1800,
        coupling_seconds=7200,
        dtype=DTypePolicy(enable_x64=True),
    )
    # Final native physics uses 01:30; 1 K/hour interpolation gives 271.5 K.
    assert_allclose_compact(result["sea_surface_temperature"], 271.5)


@pytest.mark.parametrize("save_seconds", [1800, 21600])
def test_native_flux_integral_and_endpoint_match_uninterrupted_run(
    save_seconds: int,
) -> None:
    from tests.test_jcm_surface_runtime import make_native_jcm

    coupler, _, owner = make_native_jcm(coupling_seconds=21600, seaice_coupling=False)
    owner.save_interval = timedelta(seconds=save_seconds)
    initial = coupler.initial_state()
    payload = cast(JCMAtmosphereRuntimePayload, initial._component_state("ATM").payload)
    fields = initial.component("SURFACE").fields()
    applied: list[Any] = []
    advance = owner._step_function

    def capture(state: Any, forcing: Any) -> Any:
        applied.append(forcing)
        return advance(state, forcing)

    owner._step_function = capture
    constants = PhysicalConstants()
    result = step_jax_gcm_runtime(
        owner, fields, payload, constants, DTypePolicy(enable_x64=True)
    )
    before = payload.jcm_state
    native, predictions = owner.model.run_from_state_with_carry(
        before.dycore_state,
        forcing=applied[0],
        initial_physics_state=before.physics_carry,
        initial_time=before.time,
        initial_step=before.step,
        save_interval=1 / 48,
        total_time=0.25,
        output_averages=False,
    )
    physics = predictions.physics
    surface = physics["_surface_flux"]
    for name, values in {
        "sensible_heat_flux": -np.asarray(surface.shf),
        "latent_heat_flux": -np.asarray(surface.evap)
        * float(constants.latent_heat_of_vaporization)
        / 1000,
        "downward_shortwave_radiation_flux": np.asarray(physics["_shortwave_rad"].rsds),
        "net_shortwave_radiation_flux": np.asarray(physics["_shortwave_rad"].rsns),
        "downward_longwave_radiation_flux": np.asarray(surface.rlds),
    }.items():
        assert values.shape[0] == 12
        expected = np.sum(1800 * values, axis=0).T / 21600
        assert_allclose_compact(result.fields[name], expected, rtol=1e-12, atol=1e-12)
    assert_allclose_compact(
        result.fields["air_temperature"],
        np.asarray(predictions.dynamics.temperature[-1, -1]).T,
        rtol=1e-12,
    )
    endpoint = cast(JCMAtmosphereRuntimePayload, result.payload).jcm_state
    for actual, expected in zip(
        jax.tree.leaves(endpoint.dycore_state),
        jax.tree.leaves(native.dynamics),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        jax.tree.leaves(endpoint.physics_carry),
        jax.tree.leaves(native.physics),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(endpoint.time.delta.days, native.time.delta.days)
    np.testing.assert_array_equal(
        endpoint.time.delta.seconds, native.time.delta.seconds
    )
    np.testing.assert_array_equal(endpoint.step, native.step)
