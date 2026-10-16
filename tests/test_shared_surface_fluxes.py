"""Protect CESM reference diagnostics and the existing Veros five-flux API."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor.dtypes import DTypePolicy
from vercor.fluxes.bulk_formula_cesm import compute_ocean_surface_fluxes
from vercor.physics import PhysicalConstants

pytestmark = pytest.mark.fast_always


def surface_fields(shape: tuple[int, int] = (2, 3)) -> dict[str, jax.Array]:
    """Return colocated model-level inputs plus distinct 10 m wind/radiation."""
    return {
        name: jnp.full(shape, value)
        for name, value in {
            "model_level_height": 10.0,
            "u_velocity": 8.0,
            "v_velocity": 2.0,
            "potential_temperature": 299.0,
            "specific_humidity": 0.01,
            "density": 1.2,
            "temperature": 299.0,
            "sea_surface_temperature": 300.0,
            "sea_surface_salinity": 35.0,
            "ocean_u_velocity": 0.5,
            "ocean_v_velocity": 0.0,
            "sea_surface_height": 0.0,
            "net_shortwave_radiation_flux": 120.0,
            "downward_shortwave_radiation_flux": 150.0,
            "downward_longwave_radiation_flux": 250.0,
            "rain_rate": 3e-5,
            "snow_rate": 1e-5,
            "surface_pressure": 100000.0,
            "u_velocity_10m": 4.0,
            "v_velocity_10m": -3.0,
        }.items()
    }


def test_veros_reference_before_shared_extraction() -> None:
    from vercor.setups._external.veros_fluxes import compute_fluxes

    jax.config.update("jax_enable_x64", True)
    native = SimpleNamespace(
        variables=SimpleNamespace(
            tau=0,
            u=np.full((7, 6, 1, 1), 0.5),
            v=np.zeros((7, 6, 1, 1)),
            temp=np.full((7, 6, 1, 1), 26.85),
            maskT=np.ones((7, 6, 1)),
        )
    )
    result = compute_fluxes(
        native, surface_fields(), PhysicalConstants(), DTypePolicy(enable_x64=True)
    )
    # Independent literals from the unchanged CESM reference state.
    expected = [
        0.08261268009451078,
        0.022030048025202875,
        -433.5958440702862,
        55.307247258870916,
        3.229756263790135e-6,
    ]
    np.testing.assert_allclose([np.mean(v) for v in result], expected, rtol=1e-10)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_shared_cesm_diagnostics_match_reference(enable_x64: bool) -> None:
    from vercor.setups._surface_fluxes import compute_surface_fluxes

    jax.config.update("jax_enable_x64", enable_x64)
    dtype = DTypePolicy(enable_x64=enable_x64)
    fields = surface_fields()
    result = compute_surface_fluxes(
        fields,
        sea_surface_temperature=fields["sea_surface_temperature"],
        u_ocean=fields["ocean_u_velocity"],
        v_ocean=fields["ocean_v_velocity"],
        ocean_mask=jnp.ones((2, 3)),
        constants=PhysicalConstants(),
        dtype=dtype,
    )
    expected = [
        0.08261268009451078,
        0.022030048025202875,
        -433.5958440702862,
        55.307247258870916,
        -0.0001330765493887647,
        299.069981902988,
        0.01106368351883935,
    ]
    np.testing.assert_allclose(
        [np.mean(v) for v in result], expected, rtol=2e-5 if not enable_x64 else 1e-10
    )
    assert all(v.dtype == dtype.jax_real for v in result)


@pytest.mark.parametrize("height", [10.0, 100.0])
def test_reference_temperature_and_humidity_use_same_model_height(
    height: float,
) -> None:
    from vercor.setups._surface_fluxes import compute_surface_fluxes

    jax.config.update("jax_enable_x64", True)
    f = surface_fields()
    f["model_level_height"] = jnp.full((2, 3), height)
    mask = jnp.ones((2, 3))
    direct = compute_ocean_surface_fluxes(
        PhysicalConstants(),
        mask,
        f["model_level_height"],
        f["u_velocity"],
        f["v_velocity"],
        f["potential_temperature"],
        f["specific_humidity"],
        f["density"],
        f["temperature"],
        f["ocean_u_velocity"],
        f["ocean_v_velocity"],
        f["sea_surface_temperature"],
    )
    result = compute_surface_fluxes(
        f,
        sea_surface_temperature=f["sea_surface_temperature"],
        u_ocean=f["ocean_u_velocity"],
        v_ocean=f["ocean_v_velocity"],
        ocean_mask=mask,
        constants=PhysicalConstants(),
        dtype=DTypePolicy(enable_x64=True),
    )
    np.testing.assert_allclose(result.tref, direct[6], rtol=1e-12)
    np.testing.assert_allclose(result.qref, direct[7], rtol=1e-12)
    assert not np.allclose(result.tref, f["temperature"])
    assert not np.allclose(result.qref, f["specific_humidity"])
