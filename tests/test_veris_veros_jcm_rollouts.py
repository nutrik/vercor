"""Exercise authentic native first-air, current surface composition and shared output."""

from typing import Any
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from vercor.output import OutputTarget
from vercor.regridding import bilinear, conservative
from tests.test_veris_veros_jcm_gallery import native_coupler

pytestmark = [pytest.mark.slow, pytest.mark.fast_always]


def test_first_ice_step_uses_initialized_jcm(monkeypatch: pytest.MonkeyPatch) -> None:
    from vercor.setups._external import veris_runtime

    coupler = native_coupler(monkeypatch, steps=1)
    initial = coupler.initial_state()
    captured: list[Any] = []
    original = veris_runtime.prepare_veris_forcing_pack

    def capture(fields: Any, **kwargs: Any) -> Any:
        captured.append(
            {key: np.asarray(value).copy() for key, value in fields.items()}
        )
        return original(fields, **kwargs)

    monkeypatch.setattr(veris_runtime, "prepare_veris_forcing_pack", capture)
    coupler.run(initial)
    atm = coupler.components["ATM"]
    ice = coupler.components["CICE"]
    wet = np.asarray(ice.grid.binary_mask) > 0
    remap = bilinear(atm.grid, ice.grid)
    for name in (
        "downward_shortwave_radiation_flux",
        "surface_pressure",
        "temperature",
        "specific_humidity",
        "model_level_height",
    ):
        expected = remap.regrid(initial.component("ATM").field(name))
        np.testing.assert_allclose(captured[0][name][wet], np.asarray(expected)[wet])
    assert np.any(captured[0]["downward_shortwave_radiation_flux"][wet] > 0)
    assert np.all(captured[0]["surface_pressure"][wet] > 80000)


def test_native_rollout_composes_current_surfaces_and_shared_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    coupler = native_coupler(monkeypatch, steps=2, output=True)
    initial = coupler.initial_state()
    final = coupler.run(initial, output=OutputTarget(tmp_path))
    ice = final.component("CICE")
    ocean = final.component("OCN")
    atm = final.component("ATM")
    wet = jnp.asarray(ice.grid.binary_mask)
    area = jnp.asarray(ice.field("ice_fraction"))
    remap = conservative(ice.grid, atm.grid)
    land_fraction = 1 - remap.regrid(wet)
    # Apply the established static-mask threshold, as the paired terrain does.
    fo = jnp.clip(1 - land_fraction, 0, 1)
    fo = jnp.where(fo < 0.001, 0, jnp.where(fo > 0.999, 1, fo))
    fi = remap.regrid(wet * area)
    water = remap.regrid(wet * (1 - area) * ocean.field("sea_surface_temperature"))
    ice_temperature = remap.regrid(wet * area * ice.field("ice_surface_temperature"))
    ice_albedo = remap.regrid(wet * area * ice.field("ice_surface_albedo"))
    marine = np.asarray(fo) > 0
    np.testing.assert_allclose(
        atm.field("sea_surface_temperature", scope="state")[marine],
        np.asarray((water + ice_temperature) / jnp.where(fo > 0, fo, 1))[marine],
    )
    np.testing.assert_allclose(
        atm.field("marine_surface_albedo")[marine],
        np.asarray(((fo - fi) * 0.07 + ice_albedo) / jnp.where(fo > 0, fo, 1))[marine],
    )
    np.testing.assert_allclose(
        atm.field("marine_ice_fraction")[marine],
        np.asarray(fi / jnp.where(fo > 0, fo, 1))[marine],
    )
    assert (
        np.max(
            np.abs(
                np.asarray(area)
                - np.asarray(initial.component("CICE").field("ice_fraction"))
            )
        )
        > 0
    )
    # Prescribed land consumes no atmospheric fluxes in JCM 3. Its delivered
    # surface flux diagnostics are grid means, not separate land tiles.
    assert coupler.components["LND"].spec.inputs == ()
    files = sorted(tmp_path.glob("**/*.nc"))
    assert files
    atm_files = [path for path in files if path.name.startswith("atm.averages.")]
    assert atm_files
    with xr.open_dataset(atm_files[-1]) as ds:
        for name, unit in [
            ("marine_surface_albedo", "1"),
            ("marine_ice_fraction", "1"),
            ("sea_surface_temperature", "K"),
        ]:
            assert ds[name].dims[-2:] == ("lat", "lon")
            assert ds[name].attrs["units"] == unit
            np.testing.assert_allclose(ds[name].isel(time=-1), atm.field(name))
        np.testing.assert_allclose(ds.lon, atm.grid.longitude)
        np.testing.assert_allclose(ds.lat, atm.grid.latitude)

    with xr.open_dataset(atm_files[-1]) as last:
        for name, native, factor in [
            ("sensible_heat_flux", "surface_flux.shf", -1.0),
            (
                "latent_heat_flux",
                "surface_flux.evap",
                -coupler.constants.latent_heat_of_vaporization / 1000,
            ),
        ]:
            expected = last[native].isel(time=-1) * factor
            np.testing.assert_allclose(
                atm.field(name), np.asarray(expected.transpose("lat", "lon"))
            )
