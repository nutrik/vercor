from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tests.assertions import assert_allclose_compact
import vercor.diagnostics as diagnostics_module
from setups.data.era5_atmosphere import (
    _compute_monthly_diagnostics,
    _decode_surface_pressure,
)
from setups.data._field_helpers import mask_time_last_surface_field
from setups.data.era5_land import _prepare_era5_land_runtime_fields
from setups.data.era5_ocean import (
    _ocean_binary_mask_from_land_fraction,
)
from setups.data.erainterim_ocean import (
    _assemble_erainterim_field,
    _assemble_erainterim_latitude,
    _binary_ocean_mask_from_salinity,
)
from setups.data.camulator_land import (
    _prepare_camulator_land_surface_temperature,
)
from setups.data.jcm_land import (
    _jcm_coordinates_in_degrees,
    _prepare_jcm_land_runtime_fields,
)
from vercor.runtime import RuntimeFieldStore
from vercor.runtime.views import RuntimeComponentView
from vercor.settings import VercorSettings


def test_era5_atmosphere_helpers_support_jit_and_gradients() -> None:
    settings = VercorSettings()
    lnsp = jnp.log(jnp.asarray([[100000.0, 100500.0], [101000.0, 101500.0]]))
    hyai = jnp.asarray([1000.0, 2000.0, 3000.0])
    hybi = jnp.asarray([0.10, 0.20, 0.30])
    hyam = jnp.asarray([1500.0, 2500.0])
    hybm = jnp.asarray([0.15, 0.25])
    temperature_3d = jnp.asarray(
        [
            [[280.0, 282.0], [284.0, 286.0]],
            [[288.0, 290.0], [292.0, 294.0]],
        ]
    )
    specific_humidity_3d = jnp.full((2, 2, 2), 0.002)
    temperature = temperature_3d[..., 0]

    surface_pressure = jax.jit(_decode_surface_pressure)(lnsp)
    (
        model_level_height,
        density,
        potential_temperature,
    ) = jax.jit(
        lambda sp, t3d, q3d, t: _compute_monthly_diagnostics(
            settings,
            sp,
            hyai,
            hybi,
            hyam,
            hybm,
            t3d,
            q3d,
            t,
        )
    )(
        surface_pressure,
        temperature_3d,
        specific_humidity_3d,
        temperature,
    )
    combine_surface_temperatures = diagnostics_module.combine_surface_temperatures
    combined_surface_temperature = jax.jit(combine_surface_temperatures)(
        jnp.asarray([[jnp.nan, 270.0], [271.0, jnp.nan]]),
        jnp.asarray([[272.0, jnp.nan], [273.0, 274.0]]),
    )

    assert_allclose_compact(surface_pressure, np.exp(np.asarray(lnsp)))
    assert model_level_height.shape == (2, 2)
    assert density.shape == (2, 2)
    assert potential_temperature.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(model_level_height)))
    assert np.all(np.isfinite(np.asarray(density)))
    assert np.all(np.isfinite(np.asarray(potential_temperature)))
    assert_allclose_compact(
        combined_surface_temperature,
        np.asarray([[272.0, 270.0], [544.0, 274.0]]),
    )
    surface_temperature_gradient = jax.grad(
        lambda land: jnp.sum(
            combine_surface_temperatures(
                land,
                jnp.asarray([[272.0, 273.0], [274.0, 275.0]]),
            )
        )
    )(jnp.asarray([[270.0, 271.0], [272.0, 273.0]]))
    assert_allclose_compact(surface_temperature_gradient, np.ones((2, 2)))

    density_gradient = jax.grad(
        lambda sp: jnp.sum(
            _compute_monthly_diagnostics(
                settings,
                sp,
                hyai,
                hybi,
                hyam,
                hybm,
                temperature_3d,
                specific_humidity_3d,
                temperature,
            )[1]
        )
    )(surface_pressure)
    assert np.all(np.isfinite(np.asarray(density_gradient)))


def test_total_surface_temperature_diagnostic_uses_runtime_view_fields() -> None:
    view = RuntimeComponentView(
        name="ATM",
        grid=None,  # type: ignore[arg-type]
        incoming=RuntimeFieldStore.from_mapping(
            {
                "land_surface_temperature": jnp.asarray(
                    [[jnp.nan, 270.0], [271.0, jnp.nan]]
                ),
                "sea_surface_temperature": jnp.asarray(
                    [[272.0, jnp.nan], [273.0, 274.0]]
                ),
            }
        ),
    )

    total = diagnostics_module.total_surface_temperature(view)

    assert_allclose_compact(total, np.asarray([[272.0, 270.0], [544.0, 274.0]]))


def test_era5_land_helper_supports_jit_and_gradients() -> None:
    longitude = jnp.asarray([0.0, 120.0, 240.0])
    latitude = jnp.asarray([-30.0, 30.0])
    binary_mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    land_surface_temperature = jnp.asarray(
        [
            [[280.0], [281.0]],
            [[282.0], [283.0]],
            [[284.0], [285.0]],
        ]
    )

    (
        prepared_longitude,
        prepared_latitude,
        prepared_binary_mask,
        prepared_land_surface_temperature,
    ) = jax.jit(_prepare_era5_land_runtime_fields)(
        longitude,
        latitude,
        binary_mask,
        land_surface_temperature,
    )

    assert isinstance(prepared_longitude, jax.Array)
    assert isinstance(prepared_latitude, jax.Array)
    assert isinstance(prepared_binary_mask, jax.Array)
    assert isinstance(prepared_land_surface_temperature, jax.Array)
    assert_allclose_compact(prepared_longitude, np.asarray([0.0, 120.0, 240.0]))
    assert_allclose_compact(prepared_latitude, np.asarray([-30.0, 30.0]))
    assert_allclose_compact(prepared_binary_mask, np.asarray(binary_mask).T)
    assert_allclose_compact(
        prepared_land_surface_temperature,
        np.asarray([[[280.0, 282.0, 284.0], [281.0, 283.0, 285.0]]]),
    )

    gradient = jax.grad(
        lambda temperature: jnp.sum(
            _prepare_era5_land_runtime_fields(
                longitude,
                latitude,
                binary_mask,
                temperature,
            )[3]
        )
    )(land_surface_temperature)
    assert_allclose_compact(
        gradient, np.ones_like(np.asarray(land_surface_temperature))
    )


def test_ocean_mask_helpers_accept_jax_arrays() -> None:
    land_fraction = jnp.asarray([[1.0, 0.4], [0.0, 1.0]])
    sea_surface_temperature = jnp.asarray(
        [
            [[280.0, 281.0], [282.0, 283.0]],
            [[284.0, 285.0], [286.0, 287.0]],
        ]
    )

    binary_mask = jax.jit(_ocean_binary_mask_from_land_fraction)(land_fraction)
    masked_sst = jax.jit(mask_time_last_surface_field)(
        sea_surface_temperature,
        binary_mask,
    )

    assert isinstance(binary_mask, jax.Array)
    assert isinstance(masked_sst, jax.Array)
    assert_allclose_compact(binary_mask, np.asarray([[0.0, 0.0], [1.0, 0.0]]))
    assert np.isnan(np.asarray(masked_sst)[0, 0, 0])
    assert np.isclose(np.asarray(masked_sst)[0, 1, 0], 282.0)
    assert masked_sst.shape == (2, 2, 2)


def test_shared_masked_surface_field_helper_supports_jit_and_gradients() -> None:
    sea_surface_temperature = jnp.asarray(
        [
            [[280.0, 281.0], [282.0, 283.0]],
            [[284.0, 285.0], [286.0, 287.0]],
        ]
    )
    binary_mask = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])

    masked = jax.jit(mask_time_last_surface_field)(
        sea_surface_temperature,
        binary_mask,
    )

    assert isinstance(masked, jax.Array)
    assert masked.shape == (2, 2, 2)
    assert np.isnan(np.asarray(masked)[0, 0, 0])
    assert_allclose_compact(
        jnp.nan_to_num(masked, nan=-1.0),
        np.asarray([[[-1.0, 284.0], [282.0, -1.0]], [[-1.0, 285.0], [283.0, -1.0]]]),
    )

    gradient = jax.grad(
        lambda temperature: jnp.nansum(
            mask_time_last_surface_field(temperature, binary_mask)
        )
    )(sea_surface_temperature)
    assert_allclose_compact(
        gradient,
        np.asarray([[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]]),
    )


def test_erainterim_helpers_prepare_jax_backed_grid_and_masked_fields() -> None:
    latitude_template = jnp.arange(-90.0, 94.0, 4.0)
    latitude_core = jnp.asarray([-78.0, -74.0])
    core_field = jnp.ones((2, 2, 12))

    latitude = _assemble_erainterim_latitude(
        latitude_core,
        latitude_template,
        3,
        5,
    )
    salinity = _assemble_erainterim_field(core_field, 46, 3, 5)
    binary_mask = _binary_ocean_mask_from_salinity(salinity)
    sea_surface_temperature = mask_time_last_surface_field(
        _assemble_erainterim_field(core_field, 46, 3, 5, offset=273.15),
        binary_mask,
    )

    assert isinstance(latitude, jax.Array)
    assert isinstance(binary_mask, jax.Array)
    assert isinstance(sea_surface_temperature, jax.Array)
    assert latitude.shape == (46,)
    assert binary_mask.shape == (46, 2)
    assert sea_surface_temperature.shape == (12, 46, 2)
    assert_allclose_compact(latitude[3:5], np.asarray([-78.0, -74.0]))
    assert np.all(np.asarray(binary_mask[3:5, :]) == 1.0)
    assert np.all(np.asarray(binary_mask[:3, :]) == 0.0)
    assert np.isnan(np.asarray(sea_surface_temperature)[0, 0, 0])
    assert np.isclose(np.asarray(sea_surface_temperature)[0, 3, 0], 274.15)


def test_jcm_land_coordinate_helper_supports_jit() -> None:
    longitude_radians = jnp.deg2rad(jnp.asarray([0.0, 180.0]))
    latitude_radians = jnp.deg2rad(jnp.asarray([-45.0, 45.0]))

    longitude_degrees, latitude_degrees = jax.jit(_jcm_coordinates_in_degrees)(
        longitude_radians,
        latitude_radians,
    )

    assert isinstance(longitude_degrees, jax.Array)
    assert isinstance(latitude_degrees, jax.Array)
    assert_allclose_compact(longitude_degrees, np.asarray([0.0, 180.0]))
    assert_allclose_compact(latitude_degrees, np.asarray([-45.0, 45.0]))


def test_jcm_land_runtime_helper_supports_jit_and_gradients() -> None:
    longitude_radians = jnp.deg2rad(jnp.asarray([0.0, 180.0]))
    latitude_radians = jnp.deg2rad(jnp.asarray([-45.0, 45.0]))
    land_surface_temperature = jnp.asarray([[280.0, 281.0], [282.0, 283.0]])
    soil_moisture = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])

    (
        longitude_degrees,
        latitude_degrees,
        prepared_temperature,
        prepared_soil_moisture,
    ) = jax.jit(_prepare_jcm_land_runtime_fields)(
        longitude_radians,
        latitude_radians,
        land_surface_temperature,
        soil_moisture,
    )

    assert isinstance(longitude_degrees, jax.Array)
    assert isinstance(latitude_degrees, jax.Array)
    assert isinstance(prepared_temperature, jax.Array)
    assert isinstance(prepared_soil_moisture, jax.Array)
    assert_allclose_compact(longitude_degrees, np.asarray([0.0, 180.0]))
    assert_allclose_compact(latitude_degrees, np.asarray([-45.0, 45.0]))
    assert_allclose_compact(
        prepared_temperature, np.asarray(land_surface_temperature).T
    )
    assert_allclose_compact(prepared_soil_moisture, np.asarray(soil_moisture).T)

    temperature_gradient, soil_gradient = jax.grad(
        lambda temperature, soil: jnp.sum(
            _prepare_jcm_land_runtime_fields(
                longitude_radians,
                latitude_radians,
                temperature,
                soil,
            )[2]
            + _prepare_jcm_land_runtime_fields(
                longitude_radians,
                latitude_radians,
                temperature,
                soil,
            )[3]
        ),
        argnums=(0, 1),
    )(land_surface_temperature, soil_moisture)
    assert_allclose_compact(
        temperature_gradient, np.ones_like(np.asarray(land_surface_temperature))
    )
    assert_allclose_compact(soil_gradient, np.ones_like(np.asarray(soil_moisture)))


def test_camulator_land_temperature_helper_supports_jit() -> None:
    land_surface_temperature = jnp.asarray([[281.0, 282.0], [283.0, 284.0]])

    prepared_temperature = jax.jit(_prepare_camulator_land_surface_temperature)(
        land_surface_temperature
    )

    assert isinstance(prepared_temperature, jax.Array)
    assert_allclose_compact(
        prepared_temperature,
        np.asarray([[281.0, 282.0], [283.0, 284.0]]),
    )
