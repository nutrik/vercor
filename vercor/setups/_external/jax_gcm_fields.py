"""Map native SPEEDY model-level air, surface fluxes, and diagnosed phases."""

from __future__ import annotations

from functools import partial
from dataclasses import dataclass
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from vercor._numerical_safety import (
    _require_zero,
    replace_missing_nan,
    require_active_finite,
)
from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.types import RuntimeArray
from vercor.fluxes.vertical_coordinates import (
    _compute_surface_nearest_sigma_level_altitude,
    compute_sigma_pressure_levels,
)

REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0
REFERENCE_AIR_DENSITY = 1.2
REFERENCE_MODEL_LEVEL_HEIGHT = 50.0
COLD_SURFACE_TEMPERATURE_THRESHOLD = 250.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class JCMCoupledSurface:
    """Intensive native surface forcing and area-weighted total temperature.

    Arrays use canonical (latitude, longitude) layout. Marine temperature and
    albedo are per ocean area; ice concentration is normalized by ocean area.
    Land temperature and native soil availability retain their independent values.
    """

    land_surface_temperature: jax.Array
    sea_surface_temperature: jax.Array
    soil_moisture: jax.Array
    marine_surface_albedo: jax.Array
    marine_ice_fraction: jax.Array
    total_surface_temperature: jax.Array


@partial(jax.jit, static_argnames=("dtype",))
def prepare_coupled_surface_forcing(
    fields: Mapping[str, RuntimeArray],
    land_fraction: RuntimeArray,
    open_water_albedo: RuntimeArray,
    *,
    dtype: DTypePolicy,
) -> JCMCoupledSurface:
    """Compose separately remapped water/ice contributions without reweighting.

    Incoming ice properties already contain R(W*A*property); ice_fraction is
    R(W*A), and the ocean temperature contribution is R(W*(1-A)*SST).
    Static land coverage follows the existing ocean-mask helper's threshold.
    Undefined absent-surface operands are neutralized before arithmetic, and
    a finite reference temperature fills native branches on absent surfaces.
    """
    land = as_jax_real_array(land_fraction, dtype)
    require_active_finite(land, active_mask=None, owner="JCM land fraction")
    _require_zero(
        jnp.count_nonzero((land < 0) | (land > 1)), "JCM land fraction", "range [0, 1]"
    )
    ocean = 1 - land
    marine = ocean > 0
    ice = as_jax_real_array(fields["ice_fraction"], dtype)
    require_active_finite(ice, active_mask=marine, owner="JCM ice_fraction")
    ice = jnp.where(marine, ice, 0)
    _require_zero(
        jnp.count_nonzero((ice < 0) | (ice > ocean)),
        "JCM ice_fraction",
        "ocean-area bounds",
    )
    water = ocean - ice

    def supported(
        value: RuntimeArray,
        support: jax.Array,
        owner: str,
        *,
        retain_finite: bool = False,
    ) -> jax.Array:
        array = jnp.broadcast_to(as_jax_real_array(value, dtype), support.shape)
        require_active_finite(array, active_mask=support, owner=owner)
        # Weighted fields are already zero on absent surfaces. Retain their
        # finite zero operands so their limiting area derivatives survive;
        # sanitize missing inactive NaNs before they enter arithmetic.
        active = (marine & jnp.isfinite(array)) if retain_finite else support
        return jnp.where(active, array, 0)

    land_active = land > 0
    land_temperature = supported(
        fields["land_surface_temperature"], land_active, "JCM land temperature"
    )
    soil = supported(fields["soil_moisture"], land_active, "JCM soil moisture")
    water_temperature = supported(
        fields["open_water_temperature_contribution"],
        water > 0,
        "JCM open-water contribution",
        retain_finite=True,
    )
    ice_temperature = supported(
        fields["ice_surface_temperature"],
        ice > 0,
        "JCM ice temperature contribution",
        retain_finite=True,
    )
    ice_albedo = supported(
        fields["ice_surface_albedo"],
        ice > 0,
        "JCM ice albedo contribution",
        retain_finite=True,
    )
    water_albedo = supported(
        open_water_albedo, water > 0, "JCM open-water albedo", retain_finite=True
    )
    denominator = jnp.where(marine, ocean, 1)
    temperature = (water_temperature + ice_temperature) / denominator
    albedo = (water * water_albedo + ice_albedo) / denominator
    reference = as_jax_real_array(REFERENCE_SURFACE_TEMPERATURE, dtype)
    land_temperature = jnp.where(land_active, land_temperature, reference)
    temperature = jnp.where(marine, temperature, reference)
    return JCMCoupledSurface(
        land_temperature,
        temperature,
        soil,
        albedo,
        ice / denominator,
        land * land_temperature + ocean * temperature,
    )


JAXGCM_INPUT_GRID_FIELD_NAMES = (
    "land_surface_temperature",
    "sea_surface_temperature",
    "soil_moisture",
)
JAXGCM_COUPLED_INPUT_GRID_FIELD_NAMES = (
    "land_surface_temperature",
    "soil_moisture",
    "open_water_temperature_contribution",
    "ice_surface_temperature",
    "ice_surface_albedo",
    "ice_fraction",
)
JAXGCM_COUPLED_OUTPUT_GRID_FIELD_NAMES = (
    "marine_surface_albedo",
    "marine_ice_fraction",
)
JAXGCM_OUTPUT_GRID_FIELD_NAMES = (
    "downward_shortwave_radiation_flux",
    "surface_pressure",
    "open_water_albedo",
    "land_sensible_heat_flux",
    "land_latent_heat_flux",
    "u_velocity",
    "v_velocity",
    "temperature",
    "specific_humidity",
    "sensible_heat_flux",
    "latent_heat_flux",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "density",
    "potential_temperature",
    "model_level_height",
    "rain_rate",
    "snow_rate",
)
JAXGCM_REQUIRED_GRID_FIELD_NAMES = (
    *JAXGCM_INPUT_GRID_FIELD_NAMES,
    "total_surface_temperature",
    *JAXGCM_OUTPUT_GRID_FIELD_NAMES,
)


@jax.jit
def cleanup_surface_temperature_fields(
    land_surface_temperature: object,
    sea_surface_temperature: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    land_surface_temperature_array = replace_missing_nan(
        as_jax_real_array(land_surface_temperature),
        owner="JCM land surface temperature",
    )
    sea_surface_temperature_array = replace_missing_nan(
        as_jax_real_array(sea_surface_temperature),
        owner="JCM sea surface temperature",
    )
    total_surface_temperature = (
        land_surface_temperature_array + sea_surface_temperature_array
    )
    cold_surface_cells = total_surface_temperature < COLD_SURFACE_TEMPERATURE_THRESHOLD
    return (
        land_surface_temperature_array,
        sea_surface_temperature_array,
        total_surface_temperature,
        cold_surface_cells,
    )


@jax.jit
def prepare_surface_temperature_forcing(
    total_surface_temperature: object,
    land_fraction_mask: object,
) -> tuple[jax.Array, jax.Array]:
    """Build finite JCM land and sea temperatures from one surface field.

    JCM applies ``land_fraction_mask`` when combining its land and sea fluxes,
    so fractional coastal cells require the complete temperature in both
    forcing arrays. Scaling a temperature by its surface fraction is
    physically invalid and can make JCM's reverse-mode surface-flux derivative
    non-finite. A reference value fills only cells where the corresponding
    surface is absent.
    """

    total_surface_temperature_array = as_jax_real_array(total_surface_temperature)
    land_fraction_mask_array = as_jax_real_array(land_fraction_mask)

    land_surface_temperature = jnp.where(
        land_fraction_mask_array > 0.0,
        total_surface_temperature_array,
        REFERENCE_SURFACE_TEMPERATURE,
    )
    sea_surface_temperature = jnp.where(
        land_fraction_mask_array < 1.0,
        total_surface_temperature_array,
        REFERENCE_SURFACE_TEMPERATURE,
    )

    return land_surface_temperature, sea_surface_temperature


@partial(jax.jit, static_argnames=("dtype",))
def map_jcm_output_fields(
    latvap: float,
    reference_pressure: float,
    sigma_levels: object,
    mwdair: float,
    rgas: float,
    potential_temperature_reference_pressure: float,
    cappa: float,
    surface_sensible_heat_flux: object,
    surface_evaporation: object,
    downward_longwave_radiation_flux: object,
    net_shortwave_radiation_flux: object,
    normalized_surface_pressure: object,
    u_wind: object,
    v_wind: object,
    temperature: object,
    specific_humidity: object,
    *,
    downward_shortwave_radiation_flux: object,
    open_water_albedo: object,
    large_scale_precipitation: object,
    convective_precipitation: object,
    temperature_2m: object,
    snow_temperature: object,
    rain_temperature: object,
    dtype: DTypePolicy,
) -> dict[str, jax.Array]:
    """Map native SPEEDY fields to the coupler's (latitude, longitude) grid.

    ``precls`` and ``precnv`` are downward-positive g/m²/s. Sum them before
    regridding and convert once to kg/m²/s. On the native (longitude,
    latitude) grid, partition the sum using near-surface ``surface_flux.t0``
    (``temperature_2m``, K), then transpose both phases for exchange. The
    snow fraction is sigmoid(4 * (midpoint - T) / threshold_span), matching
    the midpoint and midpoint slope of the calibrated linear-transition
    model in Bedard-Therrien et al. (2025), doi:10.5194/hess-29-1135-2025,
    Appendix C. The ordered thresholds are in K with a positive span.
    The sigmoid retains phase tails outside that interval; it is a smooth
    approximation, not the paper's piecewise-linear model or a new fit.

    SPEEDY's t0 is a near-surface diagnostic without a specified 2 m height.
    All exported air fields retain the bottom sigma-center meaning required
    by the bulk fluxes and ``model_level_height``; t0 is used only for phase
    diagnosis. Radiation and turbulent heat fluxes are surface quantities.
    """

    latvap_array = as_jax_real_array(latvap, dtype)
    reference_pressure_array = as_jax_real_array(reference_pressure, dtype)
    sigma_levels_array = as_jax_real_array(sigma_levels, dtype)
    mwdair_array = as_jax_real_array(mwdair, dtype)
    rgas_array = as_jax_real_array(rgas, dtype)
    potential_temperature_reference_pressure_array = as_jax_real_array(
        potential_temperature_reference_pressure, dtype
    )
    cappa_array = as_jax_real_array(cappa, dtype)
    normalized_surface_pressure_array = as_jax_real_array(
        normalized_surface_pressure, dtype
    ).T
    temperature_array = as_jax_real_array(temperature, dtype)
    specific_humidity_array = as_jax_real_array(specific_humidity, dtype)

    u_velocity_bottom = as_jax_real_array(u_wind, dtype)[-1, :, :].T
    v_velocity_bottom = as_jax_real_array(v_wind, dtype)[-1, :, :].T
    temperature_bottom = temperature_array[-1, :, :].T
    specific_humidity_bottom = specific_humidity_array[-1, :, :].T / 1000.0
    temperature_2m = as_jax_real_array(temperature_2m, dtype)
    snow_temperature_array = as_jax_real_array(snow_temperature, dtype)
    rain_temperature_array = as_jax_real_array(rain_temperature, dtype)
    total_precipitation = (
        as_jax_real_array(large_scale_precipitation, dtype)
        + as_jax_real_array(convective_precipitation, dtype)
    ) / 1000.0
    threshold_span = rain_temperature_array - snow_temperature_array
    midpoint = 0.5 * (rain_temperature_array + snow_temperature_array)
    # A logistic has midpoint derivative -k/4; LT has derivative -1/span.
    snow_fraction = jax.nn.sigmoid(4.0 * (midpoint - temperature_2m) / threshold_span)
    snow_precipitation = total_precipitation * snow_fraction
    rain_rate = (total_precipitation - snow_precipitation).T
    snow_rate = snow_precipitation.T

    # SPEEDY stores land, marine, then the area-combined channel.
    sensible_channels = as_jax_real_array(surface_sensible_heat_flux, dtype)
    latent_channels = as_jax_real_array(surface_evaporation, dtype) * (
        latvap_array / 1000
    )
    sensible_heat_flux = -sensible_channels[:, :, 2].T
    latent_heat_flux = -latent_channels[:, :, 2].T
    net_shortwave_radiation_flux_surface = as_jax_real_array(
        net_shortwave_radiation_flux, dtype
    ).T
    downward_longwave_radiation_flux_surface = as_jax_real_array(
        downward_longwave_radiation_flux, dtype
    ).T

    pressure = compute_sigma_pressure_levels(
        reference_pressure_array,
        as_jax_real_array(0.0, dtype),
        sigma_levels_array,
        normalized_surface_pressure_array,
    )

    pressure_bottom = pressure[-1, :, :]
    density_bottom = mwdair_array / rgas_array * pressure_bottom / temperature_bottom
    potential_temperature_bottom = (
        temperature_bottom
        * (potential_temperature_reference_pressure_array / pressure_bottom)
        ** cappa_array
    )

    model_level_height = _compute_surface_nearest_sigma_level_altitude(
        temperature_bottom,
        sigma_levels_array[-1],
        specific_humidity_bottom,
    )

    return {
        "land_sensible_heat_flux": -sensible_channels[:, :, 0].T,
        "land_latent_heat_flux": -latent_channels[:, :, 0].T,
        "downward_shortwave_radiation_flux": jnp.broadcast_to(
            as_jax_real_array(downward_shortwave_radiation_flux, dtype),
            temperature_bottom.T.shape,
        ).T,
        "surface_pressure": reference_pressure_array
        * normalized_surface_pressure_array,
        "open_water_albedo": jnp.broadcast_to(
            as_jax_real_array(open_water_albedo, dtype), temperature_bottom.shape
        ),
        "u_velocity": u_velocity_bottom,
        "v_velocity": v_velocity_bottom,
        "temperature": temperature_bottom,
        "specific_humidity": specific_humidity_bottom,
        "sensible_heat_flux": sensible_heat_flux,
        "latent_heat_flux": latent_heat_flux,
        "net_shortwave_radiation_flux": net_shortwave_radiation_flux_surface,
        "downward_longwave_radiation_flux": downward_longwave_radiation_flux_surface,
        "pressure": pressure,
        "density": density_bottom,
        "potential_temperature": potential_temperature_bottom,
        "model_level_height": model_level_height,
        "rain_rate": rain_rate,
        "snow_rate": snow_rate,
    }


__all__ = [
    "COLD_SURFACE_TEMPERATURE_THRESHOLD",
    "JAXGCM_INPUT_GRID_FIELD_NAMES",
    "JAXGCM_OUTPUT_GRID_FIELD_NAMES",
    "JAXGCM_REQUIRED_GRID_FIELD_NAMES",
    "REFERENCE_AIR_DENSITY",
    "REFERENCE_MODEL_LEVEL_HEIGHT",
    "REFERENCE_SURFACE_TEMPERATURE",
    "cleanup_surface_temperature_fields",
    "map_jcm_output_fields",
    "prepare_surface_temperature_forcing",
]
