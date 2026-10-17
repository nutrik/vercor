"""Validate total sea-ice ocean feedback using actual native tracer kernels."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._veros_surface_support import feedback_fields, surface_model
from vercor.dtypes import DTypePolicy
from vercor.physics import PhysicalConstants
from vercor.setups import VerosConfig

pytestmark = pytest.mark.fast_always


@pytest.fixture(scope="module")
def host_backend() -> None:
    """Configure the supported stock lane before importing native operators."""
    pytest.importorskip("veros")
    from vercor.setups._external.veros_runtime_settings import configure_veros_runtime

    configure_veros_runtime("host")


def test_coupled_mode_is_explicit_and_preserves_legacy_positionals() -> None:
    config = VerosConfig("SEA", "global_4deg", "host", None, True, False)
    assert config.restore_to_climatology_temp and config.surface_forcing == "atmosphere"
    assert VerosConfig(surface_forcing="coupled").surface_forcing == "coupled"


@pytest.mark.parametrize(
    "mode,error", [("unknown", ValueError), (None, TypeError), (True, TypeError)]
)
def test_invalid_surface_forcing_mode_is_rejected(
    mode: Any, error: type[Exception]
) -> None:
    with pytest.raises(error, match="surface_forcing"):
        VerosConfig(surface_forcing=mode)


@pytest.mark.parametrize(
    "kwargs",
    [dict(setup="acc"), dict(custom_parameters={"enable_streamfunction": True})],
)
def test_coupled_mode_rejects_incompatible_native_execution(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="coupled|streamfunction"):
        VerosConfig(surface_forcing="coupled", **kwargs)


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize(
    "mass,salt,expected", [(1e-4, 0.0, -3.47e-6), (-1e-4, -5e-7, 2.97e-6)]
)
def test_total_mass_and_actual_salt_conversion(
    enable_x64: bool, mass: float, salt: float, expected: float, host_backend: None
) -> None:
    from vercor.setups._external.veros_fluxes import compute_coupled_fluxes

    jax.config.update("jax_enable_x64", True)
    policy = DTypePolicy(enable_x64=enable_x64)
    fields = feedback_fields(mass=mass, salt=salt)
    fields["sea_surface_salinity"] = jnp.zeros((4, 4))
    taux, tauy, qnet, virtual = compute_coupled_fluxes(
        fields, PhysicalConstants(), policy
    )
    np.testing.assert_allclose(
        virtual, expected, rtol=2e-6 if not enable_x64 else 1e-12
    )
    np.testing.assert_array_equal(qnet, -20.0)
    assert all(value.dtype == policy.jax_real for value in (taux, tauy, qnet, virtual))


@pytest.mark.parametrize("restore_temp", [False, True])
@pytest.mark.parametrize("restore_salt", [False, True])
@pytest.mark.parametrize("dt,depth", [(600.0, 10.0), (300.0, 25.0)])
def test_total_feedback_enters_native_tracers_once(
    restore_temp: bool, restore_salt: bool, dt: float, depth: float, host_backend: None
) -> None:
    from vercor.setups._external.veros_fluxes import compute_coupled_fluxes
    from vercor.setups._external.veros_state import (
        prepare_surface_forcing_fields,
        apply_veros_forcing_fields,
    )
    from veros.core import thermodynamics

    model = surface_model(
        restore_temp=restore_temp, restore_salt=restore_salt, dt=dt, top_depth=depth
    )
    state = model.state
    vs = state.variables
    original_qnec = np.asarray(vs.qnec).copy()
    assert float(vs.coupled_feedback_active) == 0.0
    fluxes = compute_coupled_fluxes(
        feedback_fields(), PhysicalConstants(), DTypePolicy(enable_x64=True)
    )
    taux, tauy, heat, salt = fluxes
    prepared = prepare_surface_forcing_fields(taux, tauy, heat, None, salt)
    state = apply_veros_forcing_fields(
        state, prepared, execution="host", activate_coupled_feedback=True
    )
    np.testing.assert_array_equal(state.variables.qnec, original_qnec)
    model.set_forcing(state)
    vs = state.variables
    wet = np.asarray(vs.maskT[2:-2, 2:-2, -1]) > 0
    capacity = state.settings.rho_0 * 3991.86795711963
    expected_heat = -20.0 + (8.0 if restore_temp else 0.0)
    expected_salt = -3.47e-6 + (depth / (30 * 86400.0) if restore_salt else 0.0)
    np.testing.assert_allclose(
        vs.forc_temp_surface[2:-2, 2:-2], wet * expected_heat / capacity, rtol=1e-12
    )
    np.testing.assert_allclose(
        vs.forc_salt_surface[2:-2, 2:-2], wet * expected_salt, rtol=1e-12
    )
    original_temp = np.asarray(vs.temp).copy()
    original_salt = np.asarray(vs.salt).copy()
    thermodynamics.thermodynamics(state)
    np.testing.assert_allclose(
        np.asarray(
            vs.temp[2:-2, 2:-2, -1, vs.taup1] - original_temp[2:-2, 2:-2, -1, vs.tau]
        ),
        dt * expected_heat / (capacity * depth) * wet,
        rtol=1e-11,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(
            vs.salt[2:-2, 2:-2, -1, vs.taup1] - original_salt[2:-2, 2:-2, -1, vs.tau]
        ),
        dt * expected_salt / depth * wet,
        rtol=1e-11,
        atol=1e-14,
    )
    # Re-evaluation assigns salt flux; it cannot accumulate the previous value.
    model.set_forcing(state)
    np.testing.assert_allclose(
        vs.forc_salt_surface[2:-2, 2:-2], wet * expected_salt, rtol=1e-12
    )


def test_inactive_spinup_retains_native_cold_proxy(host_backend: None) -> None:
    model = surface_model()
    state = model.state
    with state.variables.unlock():
        state.variables.qnet = np.full_like(state.variables.qnet, -20.0)
        state.variables.coupled_salt_flux = np.full_like(
            state.variables.coupled_salt_flux, -3.47e-6
        )
    model.set_forcing(state)
    np.testing.assert_array_equal(state.variables.forc_temp_surface, 0.0)
    np.testing.assert_array_equal(state.variables.forc_salt_surface, 0.0)


def test_native_surface_exports_keep_units_time_levels_and_real_ssh(
    host_backend: None,
) -> None:
    from vercor.setups._external.veros_state import extract_veros_runtime_surface_fields

    model = surface_model()
    state = model.state
    vs = state.variables
    with vs.unlock():
        vs.u = 100.0 + np.arange(vs.u.size).reshape(vs.u.shape)
        vs.v = 1000.0 + np.arange(vs.v.size).reshape(vs.v.shape)
        vs.ssh = 20.0 + np.arange(vs.ssh.size).reshape(vs.ssh.shape)
        vs.psi = np.full_like(vs.psi, 1e8)
    result = extract_veros_runtime_surface_fields(state, coupled=True)
    wet = np.asarray(vs.maskT[2:-2, 2:-2, -1]).T
    np.testing.assert_array_equal(result["sea_surface_temperature"], -2 * wet + 273.15)
    np.testing.assert_array_equal(result["sea_surface_salinity"], 35 * wet)
    for name, native, mask in (
        ("ocean_u_velocity", vs.u, vs.maskU),
        ("ocean_v_velocity", vs.v, vs.maskV),
    ):
        np.testing.assert_array_equal(
            result[name], native[2:-2, 2:-2, -1, vs.tau].T * mask[2:-2, 2:-2, -1].T
        )
    np.testing.assert_array_equal(
        result["sea_surface_height"], vs.ssh[2:-2, 2:-2].T * wet
    )
    assert set(extract_veros_runtime_surface_fields(state, coupled=False)) == {
        "sea_surface_temperature"
    }


def test_zero_total_feedback_activates_and_preserves_climatological_coefficient(
    host_backend: None,
) -> None:
    from vercor.setups._external.veros_state import (
        prepare_surface_forcing_fields,
        apply_veros_forcing_fields,
    )

    model = surface_model()
    state = model.state
    zero = jnp.zeros((4, 4))
    coefficient = np.asarray(state.variables.qnec).copy()
    forcing = prepare_surface_forcing_fields(zero, zero, zero, None, zero)
    assert float(state.variables.coupled_feedback_active) == 0.0
    updated = apply_veros_forcing_fields(
        state, forcing, execution="host", activate_coupled_feedback=True
    )
    assert float(updated.variables.coupled_feedback_active) == 1.0
    np.testing.assert_array_equal(updated.variables.qnec, coefficient)


def test_public_coupled_runtime_uses_only_total_fluxes(
    monkeypatch: pytest.MonkeyPatch, host_backend: None
) -> None:
    from datetime import datetime
    from tests._veros_surface_support import surface_setup_class
    from vercor import Coupler, Clock, Exchange, RuntimeOptions
    from vercor.components import DataComponent
    from vercor.recipes import VERIS_TO_OCEAN_FLUX_FIELDS, OCEAN_TO_VERIS_FORCING_FIELDS
    from vercor.fields import _flatten_field_items
    from vercor.setups import make_veros_gcm
    from vercor.setups._external import veros_gcm_state, veros_fluxes

    setup = surface_setup_class()
    monkeypatch.setattr(
        veros_gcm_state, "_select_veros_setup_class", lambda *args: setup
    )

    def forbidden_bulk(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("atmospheric bulk must not run in total-feedback mode")

    monkeypatch.setattr(veros_fluxes, "compute_fluxes", forbidden_bulk)
    ocean = make_veros_gcm(config=VerosConfig(surface_forcing="coupled"))
    assert tuple(ocean.spec.inputs) == tuple(
        _flatten_field_items(VERIS_TO_OCEAN_FLUX_FIELDS)
    )
    assert tuple(ocean.spec.outputs) == tuple(
        _flatten_field_items(OCEAN_TO_VERIS_FORCING_FIELDS)
    )
    source = DataComponent("CICE", ocean.grid, feedback_fields())
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 1200.0, 1),
        components=(source, ocean),
        exchanges=(
            Exchange(
                "CICE",
                "OCN",
                VERIS_TO_OCEAN_FLUX_FIELDS,
                quantity="source_flux",
                source_coverage="binary",
            ),
        ),
        run_order=("CICE", "OCN"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )
    initial = coupler.initial_state()
    final = coupler.run(initial)
    native = final._component_state("OCN").payload
    original_native = initial._component_state("OCN").payload
    assert native is not None and original_native is not None
    wet = np.asarray(ocean.grid.binary_mask) > 0
    np.testing.assert_allclose(
        np.asarray(final.component("OCN").field("sea_surface_salinity"))[wet],
        35.0 - 1200 * 3.47e-6 / 10,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(final.component("OCN").field("sea_surface_temperature"))[wet],
        271.15 - 1200 * 20 / (10 * native.settings.rho_0 * 3991.86795711963),
        rtol=1e-11,
    )
    assert float(native.variables.coupled_feedback_active) == 1.0
    assert float(original_native.variables.coupled_feedback_active) == 0.0


@pytest.mark.parametrize("rho", [997.0, 1030.0])
def test_fixed_reference_conversion_density_ad_and_inactive_nan(
    rho: float, host_backend: None
) -> None:
    from dataclasses import replace
    from vercor.setups._external.veros_fluxes import compute_coupled_fluxes

    fields = feedback_fields(salt=1e-7)
    for name in fields:
        fields[name] = fields[name].at[0, 0].set(jnp.nan)
    policy = DTypePolicy(enable_x64=True)

    def total(values: jax.Array) -> jax.Array:
        constants = replace(
            PhysicalConstants(),
            freshwater_density=values[0],
            surface_salinity_reference=values[1],
        )
        return jnp.sum(compute_coupled_fluxes(fields, constants, policy)[3])

    x = jnp.asarray([rho, 34.7])
    value = jax.jit(total)(x)
    expected = 15 * (-34.7e-4 + 1e-4) / rho
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    analytic = jnp.asarray([-expected / rho, -15e-4 / rho])
    np.testing.assert_allclose(jax.jacfwd(total)(x), analytic, rtol=1e-12)
    np.testing.assert_allclose(jax.grad(total)(x), analytic, rtol=1e-12)
    finite = jnp.stack(
        [
            (total(x + step * axis) - total(x - step * axis)) / (2 * step)
            for step, axis in zip((0.01, 0.001), jnp.eye(2), strict=True)
        ]
    )
    np.testing.assert_allclose(finite, analytic, rtol=1e-8)
