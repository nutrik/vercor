"""Preserve native marine forcing beyond a coupled ocean's geometric domain."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import DataComponent
from vercor.dtypes import DTypePolicy
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.regridding import conservative
from vercor.setups import JCMAtmosphereConfig, JCMLandAtmosphereConfig, JCMInputs
from vercor.topology import SurfaceMaskPolicy
from tests.test_jcm_surface_forcing import native_case

pytestmark = pytest.mark.fast_always


@pytest.mark.parametrize("enable_x64", [False, True])
def test_coupled_numerators_exclude_prescribed_marine_area(enable_x64: bool) -> None:
    from vercor.setups._external.jax_gcm_fields import prepare_coupled_surface_forcing

    policy = DTypePolicy(enable_x64=enable_x64)
    fields = {
        "land_surface_temperature": jnp.array([[300.0]]),
        "soil_moisture": jnp.array([[0.5]]),
        "open_water_temperature_contribution": jnp.array([[0.25 * 290]]),
        "ice_surface_temperature": jnp.array([[0.25 * 260]]),
        "ice_surface_albedo": jnp.array([[0.25 * 0.7]]),
        "ice_fraction": jnp.array([[0.25]]),
    }
    actual = prepare_coupled_surface_forcing(
        fields,
        jnp.array([[0.25]]),
        jnp.array(0.1),
        prescribed_marine_fraction=jnp.array([[0.25]]),
        dtype=policy,
    )
    # C=1/2, U=1/4, L=1/4. Returned marine fields are coupled
    # numerators divided by total marine area, ready for native blending.
    np.testing.assert_allclose(actual.sea_surface_temperature, 137.5 / 0.75)
    np.testing.assert_allclose(actual.marine_ice_fraction, 1 / 3)
    np.testing.assert_allclose(actual.marine_surface_albedo, 0.2 / 0.75)
    assert actual.sea_surface_temperature.dtype == policy.jax_real

    def loss(temperature: jax.Array) -> jax.Array:
        updated = {**fields, "ice_surface_temperature": 0.25 * temperature}
        result = prepare_coupled_surface_forcing(
            updated,
            jnp.array([[0.25]]),
            jnp.array(0.1),
            prescribed_marine_fraction=jnp.array([[0.25]]),
            dtype=policy,
        )
        return cast(jax.Array, result.sea_surface_temperature.sum())

    point = jnp.array(260.0, dtype=policy.jax_real)
    _, forward = jax.jvp(jax.jit(loss), (point,), (jnp.ones_like(point),))
    reverse = jax.grad(loss)(point)
    np.testing.assert_allclose([forward, reverse], 1 / 3, rtol=2e-7)


def test_coupled_ice_cannot_occupy_prescribed_ocean_area() -> None:
    from vercor.setups._external.jax_gcm_fields import prepare_coupled_surface_forcing
    from tests.test_jcm_surface_fields import fields_for

    with pytest.raises(Exception, match="ocean-area bounds"):
        prepare_coupled_surface_forcing(
            fields_for(0.75, 0.6),
            jnp.full((2, 3), 0.25),
            jnp.array(0.1),
            prescribed_marine_fraction=jnp.full((2, 3), 0.25),
            dtype=DTypePolicy(enable_x64=True),
        ).sea_surface_temperature.block_until_ready()


def test_float32_prescribed_endpoint_survives_fraction_roundoff() -> None:
    from vercor.setups._external.jax_gcm_fields import prepare_coupled_surface_forcing
    from tests.test_jcm_surface_fields import fields_for

    fields = fields_for(0.0, 0.0)
    actual = prepare_coupled_surface_forcing(
        fields,
        jnp.full((2, 3), 0.8),
        jnp.array(0.1),
        prescribed_marine_fraction=jnp.full((2, 3), 1 - 0.8),
        dtype=DTypePolicy(enable_x64=False),
    )
    np.testing.assert_array_equal(actual.sea_surface_temperature, 0.0)
    np.testing.assert_array_equal(actual.marine_surface_albedo, 0.0)
    np.testing.assert_array_equal(actual.marine_ice_fraction, 0.0)


@pytest.mark.parametrize("series", [False, True])
def test_native_time_selection_and_albedo_use_original_background(series: bool) -> None:
    from jcm.date import DateData
    from jcm.forcing import BY_DATE_INTERP, make_time_series
    from jcm.physics.speedy.speedy_terms import SpeedyForcing
    from vercor.setups._external.jax_gcm_surface import (
        CoupledJCMForcing,
        CoupledSpeedyForcing,
        blend_jcm_forcing_field,
    )
    import jax_datetime as jdt

    coords, terrain, state, native = native_case(resolution=21)
    shape = terrain.fmask.shape
    if series:
        times = jdt.to_datetime(
            np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[s]")
        )
        native = native.copy(
            sea_surface_temperature=make_time_series(
                jnp.stack([jnp.full(shape, t) for t in (270.0, 282.0)]),
                times,
                align_mode=BY_DATE_INTERP,
            ),
            sice_am=make_time_series(
                jnp.stack([jnp.full(shape, a) for a in (0.2, 0.8)]),
                times,
                align_mode=BY_DATE_INTERP,
            ),
        )
    else:
        native = native.copy(
            sea_surface_temperature=jnp.full(shape, 276.0), sice_am=jnp.full(shape, 0.5)
        )
    fraction = jnp.broadcast_to(jnp.array([0.0, 0.25, 1.0])[:, None], (3, shape[1]))
    fraction = jnp.resize(fraction, shape)
    coupled = native.copy(
        sea_surface_temperature=blend_jcm_forcing_field(
            native.sea_surface_temperature, (1 - fraction) * 290, fraction
        ),
        sice_am=blend_jcm_forcing_field(native.sice_am, (1 - fraction) * 0.3, fraction),
    )
    wrapper = CoupledJCMForcing(
        coupled,
        (1 - fraction) * 0.7,
        native_background=native,
        prescribed_marine_fraction=fraction,
    )
    term, reference = CoupledSpeedyForcing(), SpeedyForcing()
    term.cache_coords(coords)
    reference.cache_coords(coords)
    for hour in (0, 12, 24):
        date = DateData.set_date(
            jdt.to_datetime("2000-01-01") + jdt.Timedelta(seconds=hour * 3600)
        )
        background = native.select(date)
        selected = wrapper.select(date)
        _, expected = reference(state, {}, background, terrain)
        _, actual = term(state, {}, selected, terrain)
        np.testing.assert_allclose(
            selected.sea_surface_temperature,
            (1 - fraction) * 290 + fraction * background.sea_surface_temperature,
        )
        np.testing.assert_allclose(
            selected.sice_am, (1 - fraction) * 0.3 + fraction * background.sice_am
        )
        np.testing.assert_allclose(
            actual["_mod_radcon"].alb_s,
            (1 - fraction) * 0.7 + fraction * expected["_mod_radcon"].alb_s,
        )
        np.testing.assert_allclose(actual["_forcing_2d"].sice_am, selected.sice_am)
    if series:
        from vercor.setups._external.jax_gcm_runtime import (
            _sample_native_surface_forcing,
        )

        sampled = _sample_native_surface_forcing(
            jdt.to_datetime("2000-01-01"),
            jnp.int32(0),
            wrapper,
            model_seconds=1800,
            save_seconds=3600,
            coupling_seconds=7200,
            dtype=DTypePolicy(enable_x64=True),
        )
        # Native physics samples at 00:30 and 01:30 for these saved frames.
        # Their mean date is 01:00: SST=270.5 K and ice=0.225.
        np.testing.assert_allclose(
            sampled["sea_surface_temperature"],
            ((1 - fraction) * 290 + fraction * 270.5).T,
        )
        np.testing.assert_allclose(
            sampled["marine_ice_fraction"], ((1 - fraction) * 0.3 + fraction * 0.225).T
        )
    assert isinstance(wrapper.copy(), CoupledJCMForcing)


def _regional_coupler(
    *, seaice: bool, backend: str, varying: bool = False
) -> tuple[Coupler, Any]:
    from vercor.setups import make_jcm_land_atmosphere

    coords, terrain, _, forcing = native_case(resolution=21)
    forcing = forcing.copy(
        stl_am=jnp.full(terrain.fmask.shape, 300.0),
        soilw_am=jnp.full(terrain.fmask.shape, 0.5),
        sea_surface_temperature=jnp.full(terrain.fmask.shape, 278.0),
    )
    if varying:
        from jcm.forcing import BY_DATE_INTERP, make_time_series
        import jax_datetime as jdt

        times = jdt.to_datetime(
            np.array(
                ["2000-01-03T12:00:00", "2000-01-04T12:00:00"], dtype="datetime64[s]"
            )
        )
        forcing = forcing.copy(
            sea_surface_temperature=make_time_series(
                jnp.stack([jnp.full(terrain.fmask.shape, t) for t in (270.0, 282.0)]),
                times,
                align_mode=BY_DATE_INTERP,
            ),
            sice_am=make_time_series(
                jnp.stack([jnp.full(terrain.fmask.shape, a) for a in (0.2, 0.8)]),
                times,
                align_mode=BY_DATE_INTERP,
            ),
        )
    ocean = RectilinearGrid(
        "regional-ocean",
        longitude=jnp.array([45.0, 135.0, 225.0, 315.0]),
        latitude=jnp.array([-30.0, 30.0]),
        longitude_edges=jnp.array([0.0, 90.0, 180.0, 270.0, 360.0]),
        latitude_edges=jnp.array([-60.0, 0.0, 60.0]),
        binary_mask=jnp.ones((2, 4)),
    )
    pair = make_jcm_land_atmosphere(
        ocean,
        inputs=JCMInputs(coords, terrain, forcing),
        config=JCMLandAtmosphereConfig(
            atmosphere=JCMAtmosphereConfig(
                seaice_coupling=seaice,
                save_interval=timedelta(minutes=60 if varying else 30),
                output=OutputSpec(period=PeriodOutput(frequency="step")),
            )
        ),
    )
    fields = (
        {
            "open_water_temperature_contribution": 0.7 * 290,
            "ice_surface_temperature": 0.3 * 260,
            "ice_surface_albedo": 0.3 * 0.7,
            "ice_fraction": 0.3,
        }
        if seaice
        else {"sea_surface_temperature": 290.0}
    )
    source = DataComponent("OCN", ocean, fields)
    coupler = Coupler(
        Clock(
            datetime(2000, 1, 3, 12),
            7200.0 if varying else 1800.0,
            2,
            calendar="gregorian",
        ),
        components=(pair.land, source, pair.atmosphere),
        run_order=("LND", "OCN", "ATM"),
        exchanges=(
            Exchange(
                "OCN",
                "ATM",
                tuple(fields),
                quantity="source_flux" if seaice else "intensive",
                regridder_factory=conservative,
            ),
            Exchange(
                "LND",
                "ATM",
                ("land_surface_temperature", "soil_moisture"),
                quantity="intensive",
            ),
        ),
        runtime=RuntimeOptions(
            dtype=DTypePolicy(enable_x64=True),
            backend=cast(Any, backend),
            topology=SurfaceMaskPolicy(),
        ),
        log_level="error",
    )
    return coupler, pair


def test_varying_native_forcing_reaches_runtime_period_and_snapshot(
    tmp_path: Path,
) -> None:
    import xarray as xr

    coupler, pair = _regional_coupler(seaice=True, backend="jax", varying=True)
    final = coupler.run(output=OutputTarget(tmp_path))
    pole = np.abs(np.asarray(pair.atmosphere.grid.latitude)) > 70
    for name, expected in (
        ("sea_surface_temperature", 271.5),
        ("marine_ice_fraction", 0.275),
        ("marine_surface_albedo", 0.21575),
    ):
        np.testing.assert_allclose(
            np.asarray(final.component("ATM").field(name))[pole], expected
        )
        for path in (
            sorted(tmp_path.glob("atm.averages*.nc"))[-1],
            tmp_path / "atm.snapshot.nc",
        ):
            with xr.open_dataset(path) as ds:
                np.testing.assert_allclose(
                    ds[name].transpose("time", "lat", "lon").values[:, pole], expected
                )


@pytest.mark.parametrize("seaice", [False, True])
@pytest.mark.parametrize("backend", ["host", "jax"])
def test_regional_ocean_preserves_native_poles_during_integration_and_output(
    seaice: bool,
    backend: str,
    tmp_path: Path,
) -> None:
    import xarray as xr

    coupler, pair = _regional_coupler(seaice=seaice, backend=backend)
    initial = coupler.initial_state()
    pole = np.abs(np.asarray(pair.atmosphere.grid.latitude)) > 70
    np.testing.assert_allclose(np.asarray(pair.terrain.fmask).T[pole], 0.25)
    np.testing.assert_allclose(
        initial.component("LND").field("land_fraction")[pole], 0.25
    )
    final = coupler.run(initial, output=OutputTarget(tmp_path))
    atmosphere = final.component("ATM")
    native = cast(Any, final._component_state("ATM").payload).jcm_state.physics
    np.testing.assert_allclose(
        np.asarray(atmosphere.field("sea_surface_temperature"))[pole], 278.0
    )
    np.testing.assert_allclose(native["_surface_flux"].tsfc.T[pole], 281.52)
    if seaice:
        np.testing.assert_allclose(
            np.asarray(atmosphere.field("marine_ice_fraction"))[pole], 0.4
        )
        np.testing.assert_allclose(
            atmosphere.field("marine_surface_albedo"), native["_mod_radcon"].alb_s.T
        )
        files = sorted(tmp_path.glob("atm.averages*.nc")) + [
            tmp_path / "atm.snapshot.nc"
        ]
        assert len(files) == 3
        for path in files:
            with xr.open_dataset(path) as dataset:
                values = (
                    dataset["marine_ice_fraction"]
                    .transpose("time", "lat", "lon")
                    .values
                )
                np.testing.assert_allclose(values[:, pole], 0.4)
                np.testing.assert_allclose(
                    dataset["sea_surface_temperature"]
                    .transpose("time", "lat", "lon")
                    .values[:, pole],
                    278.0,
                )
