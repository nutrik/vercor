"""Check native forcing units, signs, axes, reference heights and dry-cell AD."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_shared_surface_fluxes import surface_fields
from vercor.dtypes import DTypePolicy
from vercor.physics import PhysicalConstants

pytestmark = pytest.mark.fast_always


def test_veris_recipes_keep_model_and_ten_metre_winds_distinct() -> None:
    from vercor.recipes import (
        ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
        OCEAN_TO_VERIS_FORCING_FIELDS,
    )
    from vercor.fields import VectorField

    pairs = [
        (field.u, field.v)
        for field in ATMOSPHERE_TO_VERIS_FORCING_FIELDS
        if isinstance(field, VectorField)
    ]
    assert pairs == [("u_velocity", "v_velocity"), ("u_velocity_10m", "v_velocity_10m")]
    assert "downward_shortwave_radiation_flux" in ATMOSPHERE_TO_VERIS_FORCING_FIELDS
    assert "surface_pressure" in ATMOSPHERE_TO_VERIS_FORCING_FIELDS
    assert "sea_surface_salinity" in OCEAN_TO_VERIS_FORCING_FIELDS


def _fixed_bulk(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    from vercor.setups._surface_fluxes import SurfaceFluxes
    from vercor.setups._external import veris_forcing

    def compute(f: Any, **kwargs: Any) -> Any:
        captured.update(kwargs, fields=f)
        value = jnp.ones_like(f["temperature"])
        return SurfaceFluxes(
            value, value, 45 * value, value, -2e-5 * value, 268.5 * value, 0.001 * value
        )

    monkeypatch.setattr(veris_forcing, "compute_surface_fluxes", compute)


def test_native_forcing_signs_sources_units_and_transpose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.setups._external.veris_forcing import prepare_veris_forcing

    jax.config.update("jax_enable_x64", True)
    captured: dict[str, Any] = {}
    _fixed_bulk(monkeypatch, captured)
    f = surface_fields()
    f["u_velocity_10m"] = jnp.arange(6.0).reshape(2, 3)
    with jax.disable_jit():
        result = prepare_veris_forcing(
            f,
            ocean_mask=jnp.ones((2, 3)),
            constants=PhysicalConstants(),
            dtype=DTypePolicy(enable_x64=True),
        )
    expected = {
        "Qnet": -45.0,
        "Qsw": -120.0,
        "ATemp": 268.5,
        "aqh": 0.001,
        "evap": -2e-8,
        "precip": 3e-8,
        "snowfall": 1e-8,
        "SWdown": 150.0,
        "LWdown": 250.0,
        "theta": 300.0,
        "ocSalt": 35.0,
        "surfPress": 100000.0,
        "vWind": -3.0,
        "uOcean": 0.5,
        "vOcean": 0.0,
        "ssh_an": 0.0,
        "runoff": 0.0,
    }
    assert set(result) == {*expected, "uWind"}
    for name, value in expected.items():
        assert result[name].shape == (3, 2)
        np.testing.assert_allclose(result[name], value, rtol=1e-12)
    np.testing.assert_array_equal(result["uWind"], f["u_velocity_10m"].T)
    np.testing.assert_array_equal(captured["fields"]["u_velocity"], 8.0)
    np.testing.assert_array_equal(captured["fields"]["model_level_height"], 10.0)
    assert "wSpeed" not in result


def test_current_centering_is_periodic_x_and_closed_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.setups._external.veris_forcing import prepare_veris_forcing

    captured: dict[str, Any] = {}
    _fixed_bulk(monkeypatch, captured)
    f = surface_fields()
    f["ocean_u_velocity"] = jnp.array([[1.0, 3.0, 7.0], [2.0, 6.0, 14.0]])
    f["ocean_v_velocity"] = jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    with jax.disable_jit():
        result = prepare_veris_forcing(
            f,
            ocean_mask=jnp.ones((2, 3)),
            constants=PhysicalConstants(),
            dtype=DTypePolicy.from_jax_config(),
        )
    np.testing.assert_array_equal(
        captured["u_ocean"], [[4.0, 2.0, 5.0], [8.0, 4.0, 10.0]]
    )
    np.testing.assert_array_equal(
        captured["v_ocean"], [[5.0, 10.0, 15.0], [25.0, 35.0, 45.0]]
    )
    np.testing.assert_array_equal(result["uOcean"], f["ocean_u_velocity"].T)
    np.testing.assert_array_equal(result["vOcean"], f["ocean_v_velocity"].T)


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("dry_value", [0.0, float("nan")])
def test_dry_operands_are_safe_and_forcing_gradients_match(
    enable_x64: bool, dry_value: float
) -> None:
    from vercor.setups._external.veris_forcing import prepare_veris_forcing

    jax.config.update("jax_enable_x64", enable_x64)
    dtype = DTypePolicy(enable_x64=enable_x64)
    mask = jnp.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    f = {
        name: jnp.where(mask > 0, value, dry_value)
        for name, value in surface_fields().items()
    }

    def objective(delta: jax.Array) -> jax.Array:
        fields = {
            **f,
            "temperature": f["temperature"] + delta,
            "potential_temperature": f["potential_temperature"] + delta,
        }
        result = prepare_veris_forcing(
            fields, ocean_mask=mask, constants=PhysicalConstants(), dtype=dtype
        )
        return jnp.mean(result["Qnet"] + result["ATemp"])

    x = jnp.array(0.0)
    result = prepare_veris_forcing(
        f, ocean_mask=mask, constants=PhysicalConstants(), dtype=dtype
    )
    for name, value in result.items():
        assert np.isfinite(value).all(), name
        expected = (
            273.15
            if name in {"theta", "ATemp"}
            else 100000.0 if name == "surfPress" else 0.0
        )
        np.testing.assert_allclose(np.asarray(value)[np.asarray(mask.T) == 0], expected)
        assert value.dtype == dtype.jax_real
    value, forward = jax.jvp(jax.jit(objective), (x,), (jnp.ones_like(x),))
    reverse = jax.grad(objective)(x)
    eps = 1e-4 if enable_x64 else 0.01
    fd = (objective(x + eps) - objective(x - eps)) / (2 * eps)
    assert np.isfinite([value, forward, reverse, fd]).all()
    assert abs(float(forward)) > 0.1
    np.testing.assert_allclose(
        [forward, reverse], fd, rtol=1e-6 if enable_x64 else 2e-3
    )


def test_mass_conversion_uses_traced_freshwater_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.setups._external.veris_forcing import prepare_veris_forcing

    jax.config.update("jax_enable_x64", True)
    _fixed_bulk(monkeypatch, {})
    f = surface_fields()

    def objective(rho: jax.Array) -> jax.Array:
        result = prepare_veris_forcing(
            f,
            ocean_mask=jnp.ones((2, 3)),
            constants=replace(PhysicalConstants(), freshwater_density=rho),
            dtype=DTypePolicy(enable_x64=True),
        )
        return jnp.mean(result["precip"])

    with jax.disable_jit():
        np.testing.assert_allclose(
            jax.grad(objective)(jnp.array(1000.0)), -3e-11, rtol=1e-12
        )
