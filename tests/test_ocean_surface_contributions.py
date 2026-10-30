"""Check current ocean SST contributions before conservative surface assembly."""

from datetime import datetime
from inspect import Parameter, signature
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._veros_surface_support import feedback_fields, surface_setup_class
from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import DataComponent
from vercor.dtypes import DTypePolicy
from vercor.regridding import conservative
from vercor.setups import VerosConfig

pytestmark = pytest.mark.fast_always


def contribution(sst: Any, fraction: Any, wet: Any) -> jax.Array:
    from vercor.setups._external.veros_runtime import (
        open_water_temperature_contribution,
    )

    return cast(jax.Array, open_water_temperature_contribution(sst, fraction, wet))


def ocean_coupler(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True, all_ice: bool = False
) -> tuple[Coupler, Any]:
    """Build one real native column ocean with prescribed total boundary inputs."""
    from vercor.setups import make_veros_gcm
    from vercor.setups._external import veros_gcm_state
    from vercor.recipes import SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS

    jax.config.update("jax_enable_x64", True)
    setup = surface_setup_class()
    monkeypatch.setattr(
        veros_gcm_state, "_select_veros_setup_class", lambda *args: setup
    )
    ocean = make_veros_gcm(
        config=VerosConfig(
            surface_forcing="surface_fluxes",
            custom_parameters={"dt_mom": 600.0},
            export_open_water_surface=enabled,
        )
    )
    fraction = (
        jnp.ones((4, 4))
        if all_ice
        else jnp.tile(jnp.array([0.2, 0.8, 1.0, 0.0]), (4, 1))
    )
    source = DataComponent(
        "CICE",
        ocean.grid,
        {
            **feedback_fields(heat=20, mass=0, salt=0),
            "ice_fraction": fraction,
        },
    )
    routes = [
        Exchange(
            "CICE",
            "OCN",
            SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS[:-1],
            route_id="scalars",
            quantity="source_flux",
            source_coverage="binary",
            regridder_factory=conservative,
        ),
        Exchange(
            "CICE",
            "OCN",
            SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS[-1:],
            route_id="stress",
            quantity="source_flux",
            source_coverage="binary",
        ),
    ]
    if enabled:
        routes.append(
            Exchange(
                "CICE",
                "OCN",
                ("ice_fraction",),
                route_id="coverage",
                quantity="intensive",
                source_coverage="binary",
            )
        )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 600.0, 1),
        components=(source, ocean),
        exchanges=routes,
        run_order=("CICE", "OCN"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
        log_level="error",
    )
    return coupler, ocean


def test_open_water_flag_is_explicit_keyword_only_and_requires_coupled_fluxes() -> None:
    assert VerosConfig().export_open_water_surface is False
    assert (
        signature(VerosConfig).parameters["export_open_water_surface"].kind
        is Parameter.KEYWORD_ONLY
    )
    with pytest.raises(ValueError, match="surface_fluxes"):
        VerosConfig(export_open_water_surface=True)
    assert VerosConfig(
        surface_forcing="surface_fluxes", export_open_water_surface=True
    ).export_open_water_surface


@pytest.mark.parametrize("value", [1, "yes", None])
def test_open_water_flag_requires_bool(value: Any) -> None:
    with pytest.raises(TypeError, match="export_open_water_surface"):
        VerosConfig(surface_forcing="surface_fluxes", export_open_water_surface=value)


def test_correlated_open_water_temperature_is_weighted_before_averaging() -> None:
    temperature = np.array([[280.0, 300.0]])
    fraction = np.array([[0.2, 0.8]])
    actual = contribution(temperature, fraction, np.ones((1, 2)))
    np.testing.assert_allclose(actual, [[224.0, 60.0]])
    np.testing.assert_allclose(actual.mean(), 142.0)
    assert not np.isclose(
        float(actual.mean()), (1 - fraction.mean()) * temperature.mean()
    )
    np.testing.assert_array_equal(
        actual,
        contribution(jnp.asarray(temperature), jnp.asarray(fraction), jnp.ones((1, 2))),
    )


def test_missing_inactive_water_is_neutral_before_arithmetic_and_ad() -> None:
    wet = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    fraction = jnp.array([[0.0, jnp.nan, 1.0], [jnp.nan, 0.25, jnp.nan]])
    temperature = jnp.array([[280.0, jnp.nan, jnp.nan], [jnp.nan, 289.0, jnp.nan]])

    def loss(shift: jax.Array) -> jax.Array:
        return contribution(temperature + shift, fraction, wet).sum()

    np.testing.assert_allclose(
        contribution(temperature, fraction, wet),
        [[280.0, 0.0, 0.0], [0.0, 216.75, 0.0]],
    )
    point = jnp.array(0.0)
    _, forward = jax.jvp(jax.jit(loss), (point,), (jnp.array(1.0),))
    reverse = jax.grad(loss)(point)
    np.testing.assert_allclose(forward, 1.75)
    np.testing.assert_allclose(reverse, forward)
    finite = (loss(point + 1e-4) - loss(point - 1e-4)) / 2e-4
    np.testing.assert_allclose(reverse, finite, rtol=1e-9)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan")])
def test_active_concentration_is_validated(fraction: float) -> None:
    with pytest.raises(Exception, match="ice_fraction"):
        jax.block_until_ready(
            contribution(
                jnp.array([[280.0]]), jnp.array([[fraction]]), jnp.ones((1, 1))
            )
        )


def test_post_ocean_sst_is_weighted_before_remap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tagged native transition isolates extraction order from ocean numerics."""
    coupler, ocean = ocean_coupler(monkeypatch)
    owner = ocean.spec.lifecycle.setup.__self__
    tagged = np.tile([280.0, 300.0, 280.0, 300.0], (4, 1))

    def tagged_step(state: Any) -> Any:
        vs = state.variables
        temperature = np.array(vs.temp, copy=True)
        temperature[2:-2, 2:-2, -1, vs.tau] = tagged.T - 273.15
        with vs.unlock():
            vs.temp = temperature
        return state

    owner._step_function = tagged_step
    initial = coupler.initial_state()
    final = coupler.run(initial)
    values = final.component("OCN")
    np.testing.assert_allclose(
        values.field("sea_surface_temperature")[1, :2], [280.0, 300.0]
    )
    lon = float(ocean.grid.longitude[0])
    lat = float(ocean.grid.latitude[1])
    source = RectilinearGrid(
        "two-water-cells",
        longitude=np.array([lon, lon + 1]),
        latitude=np.array([lat]),
        longitude_edges=np.array([lon - 0.5, lon + 0.5, lon + 1.5]),
        latitude_edges=np.array([lat - 0.5, lat + 0.5]),
    )
    target = RectilinearGrid(
        "marine-cell",
        longitude=np.array([lon + 0.5]),
        latitude=np.array([lat]),
        longitude_edges=np.array([lon - 0.5, lon + 1.5]),
        latitude_edges=np.array([lat - 0.5, lat + 0.5]),
    )
    received = conservative(source, target).regrid(
        values.field("open_water_temperature_contribution")[1:2, :2]
    )
    np.testing.assert_allclose(received, 142.0)


def test_real_native_heating_exports_new_sst_and_latest_concentration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler, ocean = ocean_coupler(monkeypatch)
    initial = coupler.initial_state()
    original = initial.component("OCN").field("sea_surface_temperature")
    wet = np.asarray(ocean.grid.binary_mask) > 0
    np.testing.assert_allclose(
        initial.component("OCN").field("open_water_temperature_contribution"),
        np.where(wet, original, 0),
    )
    final = coupler.run(initial)
    fields = final.component("OCN")
    wet = np.asarray(ocean.grid.binary_mask) > 0
    latest = np.asarray(fields.field("sea_surface_temperature"))
    assert np.all(latest[wet] > np.asarray(original)[wet])
    fraction = np.tile([0.2, 0.8, 1.0, 0.0], (4, 1))
    np.testing.assert_allclose(
        fields.field("open_water_temperature_contribution"),
        np.where(wet, (1 - fraction) * latest, 0),
    )


def test_open_water_export_does_not_mask_underice_ocean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler, ocean = ocean_coupler(monkeypatch, all_ice=True)
    wet = np.asarray(ocean.grid.binary_mask).copy()
    final = coupler.run()
    final_grid = final.component("OCN").grid
    assert final_grid is not None
    np.testing.assert_array_equal(final_grid.binary_mask, wet)
    np.testing.assert_array_equal(
        final.component("OCN").field("open_water_temperature_contribution"), 0
    )
    assert np.all(
        np.asarray(final.component("OCN").field("sea_surface_temperature"))[wet > 0]
        > 250
    )


def test_disabled_export_preserves_ocean_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler, ocean = ocean_coupler(monkeypatch, enabled=False)
    assert "ice_fraction" not in ocean.spec.inputs
    assert "open_water_temperature_contribution" not in ocean.spec.outputs
    assert (
        "open_water_temperature_contribution"
        not in coupler.initial_state().component("OCN").fields()
    )
