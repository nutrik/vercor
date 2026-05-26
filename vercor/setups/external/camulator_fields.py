from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import torch

from vercor.dtypes import PrecisionPolicy, as_jax_real_array, jax_full
from vercor.fluxes.vertical_coordinates import compute_hybrid_sigma_full_level_altitudes
from vercor.host_arrays import runtime_array_to_host
from vercor.settings import VercorSettings
from vercor.setups.external.camulator_tensors import StateVariableAccessor

_CAMULATOR_RUNTIME_FIELD_NAMES = (
    "specific_humidity",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "sea_surface_temperature",
    "land_surface_temperature",
    "u_velocity",
    "v_velocity",
    "temperature",
    "potential_temperature",
    "density",
    "latent_heat_flux",
    "sensible_heat_flux",
    "model_level_height",
    "total_surface_temperature",
    "temperature_3d",
    "specific_humidity_3d",
)


def _initialize_camulator_runtime_fields(
    grid_shape: tuple[int, int],
    policy: PrecisionPolicy = None,
) -> dict[str, jax.Array]:
    """Create JAX-backed zero fields for CAMulator exchange storage."""

    zeros = jax_full(grid_shape, 0.0, policy)
    return {field_name: zeros for field_name in _CAMULATOR_RUNTIME_FIELD_NAMES}


@jax.jit
def _prepare_camulator_surface_forcing(
    sea_surface_temperature: object,
    land_surface_temperature: object,
    land_mask_coslat: object,
) -> tuple[jax.Array, jax.Array]:
    """Prepare CAMulator's rescaled surface-temperature forcing field."""

    sst = jnp.nan_to_num(as_jax_real_array(sea_surface_temperature))
    skt = jnp.nan_to_num(as_jax_real_array(land_surface_temperature))
    land_mask = as_jax_real_array(land_mask_coslat)

    total_surface_temperature = jnp.where(land_mask < 1.0, sst + skt, 283.0)
    rescaled_total_surface_temperature = (
        total_surface_temperature - jnp.nanmean(total_surface_temperature)
    ) / jnp.nanstd(total_surface_temperature)

    return total_surface_temperature, rescaled_total_surface_temperature


@jax.jit
def _prepare_camulator_dynamic_forcing_chunk(
    dynamic_forcing_values: object,
) -> jax.Array:
    """Convert xarray forcing values to CAMulator's time-major layout."""

    return as_jax_real_array(dynamic_forcing_values).transpose((1, 0, 2, 3))


@jax.jit
def _prepare_camulator_sst_input(
    rescaled_total_surface_temperature: object,
) -> jax.Array:
    """Expand a rescaled SST field to CAMulator's input tensor layout."""

    return as_jax_real_array(rescaled_total_surface_temperature)[
        jnp.newaxis, jnp.newaxis, jnp.newaxis, ...
    ]


@jax.jit
def _map_camulator_prediction_arrays(
    earth_radius: float,
    gravity: float,
    rdair: float,
    zvir: float,
    mwdair: float,
    rgas: float,
    potential_temperature_reference_pressure: float,
    cappa: float,
    stef_boltz: float,
    camulator_reference_pressure: float,
    hyai: object,
    hybi: object,
    hyam: object,
    hybm: object,
    u_wind: object,
    v_wind: object,
    surface_temperature: object,
    temperature_3d: object,
    specific_humidity_3d: object,
    net_shortwave_radiation_flux_accumulated: object,
    net_longwave_radiation_flux_accumulated: object,
    surface_pressure: object,
) -> dict[str, jax.Array]:
    """Map CAMulator tensor outputs into VerCOR runtime exchange fields."""

    hyai_array = as_jax_real_array(hyai).reshape(-1)
    hybi_array = as_jax_real_array(hybi).reshape(-1)
    hyam_array = as_jax_real_array(hyam).reshape(-1)
    hybm_array = as_jax_real_array(hybm).reshape(-1)

    u_velocity = as_jax_real_array(u_wind).squeeze()[-1, :, :]
    v_velocity = as_jax_real_array(v_wind).squeeze()[-1, :, :]
    surface_temperature_array = as_jax_real_array(surface_temperature).squeeze()
    temperature_3d_array = as_jax_real_array(temperature_3d).squeeze()
    specific_humidity_3d_array = as_jax_real_array(specific_humidity_3d).squeeze()
    temperature = temperature_3d_array[-1, ...]
    specific_humidity = specific_humidity_3d_array[-1, ...]

    net_shortwave_radiation_flux = (
        as_jax_real_array(net_shortwave_radiation_flux_accumulated).squeeze() / 21600.0
    )
    net_longwave_radiation_flux = (
        as_jax_real_array(net_longwave_radiation_flux_accumulated).squeeze() / -21600.0
    )
    downward_longwave_radiation_flux = (
        stef_boltz * surface_temperature_array**4 - net_longwave_radiation_flux
    )

    surface_pressure_array = as_jax_real_array(surface_pressure).squeeze()
    p_mid = (
        hyam_array[:, jnp.newaxis, jnp.newaxis] * camulator_reference_pressure
        + hybm_array[:, jnp.newaxis, jnp.newaxis]
        * surface_pressure_array[jnp.newaxis, :, :]
    )
    p_int = (
        hyai_array[:, jnp.newaxis, jnp.newaxis] * camulator_reference_pressure
        + hybi_array[:, jnp.newaxis, jnp.newaxis]
        * surface_pressure_array[jnp.newaxis, :, :]
    )

    temperature_for_height = temperature_3d_array.T
    humidity_for_height = specific_humidity_3d_array.T
    pressure_interfaces_for_height = p_int.T
    altitude = compute_hybrid_sigma_full_level_altitudes(
        temperature_for_height,
        humidity_for_height,
        pressure_interfaces_for_height,
        earth_radius=earth_radius,
        gravity=gravity,
        rdair=rdair,
        zvir=zvir,
    )
    model_level_height = altitude[..., 0].T

    density = mwdair / rgas * p_mid[-1, :, :] / temperature
    potential_temperature = (
        temperature
        * (potential_temperature_reference_pressure / p_mid[-1, :, :]) ** cappa
    )

    return {
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        "temperature_3d": temperature_3d_array,
        "specific_humidity_3d": specific_humidity_3d_array,
        "specific_humidity": specific_humidity,
        "temperature": temperature,
        "net_shortwave_radiation_flux": net_shortwave_radiation_flux,
        "downward_longwave_radiation_flux": downward_longwave_radiation_flux,
        "model_level_height": model_level_height,
        "density": density,
        "potential_temperature": potential_temperature,
    }


def _camulator_output_array(
    accessor: StateVariableAccessor,
    prediction_out: torch.Tensor,
    variable_name: str,
) -> object:
    """Return one inverse-transformed CAMulator output variable on the host."""

    return runtime_array_to_host(
        accessor.get_state_var(prediction_out, variable_name).cpu().numpy()
    )


def _map_camulator_prediction_to_runtime_fields(
    settings: VercorSettings,
    *,
    camulator_reference_pressure: float,
    hyai: torch.Tensor,
    hybi: torch.Tensor,
    hyam: torch.Tensor,
    hybm: torch.Tensor,
    accessor_output: StateVariableAccessor,
    state_transformer: Any,
    prediction: torch.Tensor,
) -> dict[str, jax.Array]:
    """Convert one CAMulator prediction into VerCOR runtime exchange fields."""

    prediction_out = state_transformer.inverse_transform(prediction)
    return cast(
        dict[str, jax.Array],
        _map_camulator_prediction_arrays(
            settings.earth_radius,
            settings.gravity,
            settings.rdair,
            settings.zvir,
            settings.mwdair,
            settings.rgas,
            settings.p0,
            settings.cappa,
            settings.stefBoltz,
            camulator_reference_pressure,
            runtime_array_to_host(hyai.cpu().numpy()).squeeze(),
            runtime_array_to_host(hybi.cpu().numpy()).squeeze(),
            runtime_array_to_host(hyam.cpu().numpy()).squeeze(),
            runtime_array_to_host(hybm.cpu().numpy()).squeeze(),
            _camulator_output_array(accessor_output, prediction_out, "U"),
            _camulator_output_array(accessor_output, prediction_out, "V"),
            _camulator_output_array(accessor_output, prediction_out, "TS"),
            _camulator_output_array(accessor_output, prediction_out, "T"),
            _camulator_output_array(accessor_output, prediction_out, "Qtot"),
            _camulator_output_array(accessor_output, prediction_out, "FSNS"),
            _camulator_output_array(accessor_output, prediction_out, "FLNS"),
            _camulator_output_array(accessor_output, prediction_out, "PS"),
        ),
    )


__all__ = [
    "_CAMULATOR_RUNTIME_FIELD_NAMES",
    "_initialize_camulator_runtime_fields",
    "_map_camulator_prediction_arrays",
    "_map_camulator_prediction_to_runtime_fields",
    "_prepare_camulator_dynamic_forcing_chunk",
    "_prepare_camulator_sst_input",
    "_prepare_camulator_surface_forcing",
]
