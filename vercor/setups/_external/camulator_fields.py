"""Shared CAMulator meteorology and native host field conversion boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import torch

from vercor._numerical_safety import (
    replace_missing_nan,
    require_active_finite,
    require_strictly_positive,
)
from vercor.dtypes import as_jax_real_array
from vercor.fluxes.vertical_coordinates import compute_hybrid_sigma_full_level_altitudes
from vercor._host_arrays import runtime_array_to_host
from vercor.physics import PhysicalConstants
from vercor.exceptions import ComponentError
from vercor.setups._external.camulator_tensors import StateVariableAccessor

if TYPE_CHECKING:
    from vercor.setups._external.camulator_forcing import CamulatorRuntimeCursor
    from vercor.setups._external.camulator_gcm_state import CAMulatorGCMSetupState


@jax.jit
def _map_camulator_air_fields(
    constants: PhysicalConstants,
    *,
    reference_pressure: float,
    hyai: object,
    hybi: object,
    hyam: object,
    hybm: object,
    u_wind: object,
    v_wind: object,
    temperature_3d: object,
    water_vapor_3d: object,
    surface_pressure: object,
) -> dict[str, jax.Array]:
    """Map top-to-bottom native meteorology to lowest-level SI exchange fields.

    Native thermodynamics and winds have shape (level, latitude, longitude),
    surface pressure has shape (latitude, longitude). Hybrid coefficients are
    dimensionless; the shared IFS kernel returns bottom-first geometric height.
    All physical constants remain traced, including the virtual-temperature
    correction. Host normalization and history selection belong to callers.
    """
    temperature = as_jax_real_array(temperature_3d)
    vapor = as_jax_real_array(water_vapor_3d)
    ps = as_jax_real_array(surface_pressure)
    u = as_jax_real_array(u_wind)
    v = as_jax_real_array(v_wind)
    if temperature.ndim != 3 or ps.ndim != 2:
        raise ComponentError(
            "CAMulator meteorology requires (level, lat, lon) and surface (lat, lon)."
        )
    if (
        any(value.shape != temperature.shape for value in (vapor, u, v))
        or temperature.shape[1:] != ps.shape
    ):
        raise ComponentError("CAMulator meteorology shapes do not match.")
    for name, value in (
        ("temperature", temperature),
        ("water vapor", vapor),
        ("surface pressure", ps),
        ("U", u),
        ("V", v),
    ):
        require_active_finite(value, active_mask=None, owner=f"CAMulator {name}")
    require_strictly_positive(temperature, owner="CAMulator temperature")
    require_strictly_positive(ps, owner="CAMulator surface pressure")
    mid = (
        as_jax_real_array(hyam).reshape(-1, 1, 1) * reference_pressure
        + as_jax_real_array(hybm).reshape(-1, 1, 1) * ps
    )
    interfaces = (
        as_jax_real_array(hyai).reshape(-1, 1, 1) * reference_pressure
        + as_jax_real_array(hybi).reshape(-1, 1, 1) * ps
    )
    height = compute_hybrid_sigma_full_level_altitudes(
        temperature.T,
        vapor.T,
        interfaces.T,
        earth_radius=constants.earth_radius,
        gravity=constants.gravity,
        rdair=constants.dry_air_gas_constant,
        zvir=constants.water_vapor_mass_ratio_correction,
    )[..., 0].T
    return {
        "u_velocity": u[-1],
        "v_velocity": v[-1],
        "temperature_3d": temperature,
        "specific_humidity_3d": vapor,
        "temperature": temperature[-1],
        "specific_humidity": vapor[-1],
        "surface_pressure": ps,
        "pressure": mid[-1],
        "model_level_height": height,
        "density": as_jax_real_array(
            constants.dry_air_molecular_weight
            / constants.universal_gas_constant
            * mid[-1]
            / temperature[-1]
        ),
        "potential_temperature": as_jax_real_array(
            temperature[-1]
            * (constants.reference_pressure / mid[-1]) ** constants.dry_air_kappa
        ),
    }


@jax.jit
def prepare_camulator_surface_forcing(
    sea_surface_temperature: object,
    land_surface_temperature: object,
    land_mask_coslat: object,
) -> tuple[jax.Array, jax.Array]:
    """Prepare CAMulator's rescaled surface-temperature forcing field."""

    sst = replace_missing_nan(
        as_jax_real_array(sea_surface_temperature),
        owner="CAMulator sea surface temperature",
    )
    skt = replace_missing_nan(
        as_jax_real_array(land_surface_temperature),
        owner="CAMulator land surface temperature",
    )
    land_mask = as_jax_real_array(land_mask_coslat)
    require_active_finite(
        land_mask,
        active_mask=None,
        owner="CAMulator land mask",
    )

    total_surface_temperature = jnp.where(land_mask < 1.0, sst + skt, 283.0)
    mean = jnp.mean(total_surface_temperature)
    standard_deviation = jnp.std(total_surface_temperature)
    require_strictly_positive(
        standard_deviation,
        owner="CAMulator surface-temperature standard deviation",
    )
    rescaled_total_surface_temperature = (
        total_surface_temperature - mean
    ) / standard_deviation
    require_active_finite(
        total_surface_temperature,
        active_mask=None,
        owner="CAMulator total surface temperature",
    )
    require_active_finite(
        rescaled_total_surface_temperature,
        active_mask=None,
        owner="CAMulator rescaled surface temperature",
    )

    return total_surface_temperature, rescaled_total_surface_temperature


@jax.jit
def prepare_camulator_dynamic_forcing_chunk(
    dynamic_forcing_values: object,
) -> jax.Array:
    """Convert xarray forcing values to CAMulator's time-major layout."""

    dynamic_forcing = as_jax_real_array(dynamic_forcing_values)
    require_active_finite(
        dynamic_forcing,
        active_mask=None,
        owner="CAMulator dynamic forcing",
    )
    return dynamic_forcing.transpose((1, 0, 2, 3))


@jax.jit
def prepare_camulator_sst_input(
    rescaled_total_surface_temperature: object,
) -> jax.Array:
    """Expand a rescaled SST field to CAMulator's input tensor layout."""

    return as_jax_real_array(rescaled_total_surface_temperature)[
        jnp.newaxis, jnp.newaxis, jnp.newaxis, ...
    ]


@jax.jit
def map_camulator_prediction_arrays(
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
    *,
    native_step_seconds: float,
) -> dict[str, jax.Array]:
    """Map native arrays to runtime fields, converting step energies to W/m².

    Radiation inputs are downward-positive J/m² accumulated over one native
    forecast step. The duration is independent of the coupling interval;
    downward longwave includes surface emission exactly once.
    """

    require_strictly_positive(
        native_step_seconds, owner="CAMulator native step duration"
    )

    ps_native = as_jax_real_array(surface_pressure)
    ps = ps_native.reshape(ps_native.shape[-2:])
    levels = as_jax_real_array(hyam).size
    shape = (levels, *ps.shape)
    constants = PhysicalConstants(
        earth_radius=earth_radius,
        gravity=gravity,
        dry_air_gas_constant=rdair,
        water_vapor_mass_ratio_correction=zvir,
        dry_air_molecular_weight=mwdair,
        universal_gas_constant=rgas,
        reference_pressure=potential_temperature_reference_pressure,
        dry_air_kappa=cappa,
        stefan_boltzmann_constant=stef_boltz,
    )
    air = _map_camulator_air_fields(
        constants,
        reference_pressure=camulator_reference_pressure,
        hyai=hyai,
        hybi=hybi,
        hyam=hyam,
        hybm=hybm,
        u_wind=as_jax_real_array(u_wind).reshape(shape),
        v_wind=as_jax_real_array(v_wind).reshape(shape),
        temperature_3d=as_jax_real_array(temperature_3d).reshape(shape),
        water_vapor_3d=as_jax_real_array(specific_humidity_3d).reshape(shape),
        surface_pressure=ps,
    )
    ts = as_jax_real_array(surface_temperature).reshape(ps.shape)
    return {
        **air,
        "net_shortwave_radiation_flux": as_jax_real_array(
            net_shortwave_radiation_flux_accumulated
        ).reshape(ps.shape)
        / native_step_seconds,
        "downward_longwave_radiation_flux": stef_boltz * ts**4
        + as_jax_real_array(net_longwave_radiation_flux_accumulated).reshape(ps.shape)
        / native_step_seconds,
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


def _require_camulator_vapor_channel(
    accessor: StateVariableAccessor, variable: str | None, *, owner: str
) -> str:
    """Require a vapor-specific 3D channel in one native tensor layout."""
    if not isinstance(variable, str) or not variable:
        raise ComponentError(
            f"{owner} requires water_vapor_variable declaring specific water vapor."
        )
    if variable.casefold() == "qtot":
        raise ComponentError(f"{owner}: Qtot is total water, not specific water vapor.")
    try:
        index = accessor.get_var_index(variable)
        index.require_available(tensor_type=accessor.tensor_type, var_name=variable)
        if not index.is_3d:
            raise ValueError("water vapor must be a three-dimensional native variable")
    except ValueError as exc:
        raise ComponentError(f"{owner} water vapor capability: {exc}") from exc
    return variable


def map_camulator_prediction_to_runtime_fields(
    constants: PhysicalConstants,
    *,
    camulator_reference_pressure: float,
    native_step_seconds: float,
    water_vapor_variable: str | None,
    hyai: torch.Tensor,
    hybi: torch.Tensor,
    hyam: torch.Tensor,
    hybm: torch.Tensor,
    accessor_output: StateVariableAccessor,
    state_transformer: Any,
    prediction: torch.Tensor,
) -> dict[str, jax.Array]:
    """Map a native prediction using its forecast-step duration in seconds."""

    require_strictly_positive(
        native_step_seconds, owner="CAMulator native step duration"
    )
    water_vapor_variable = _require_camulator_vapor_channel(
        accessor_output, water_vapor_variable, owner="CAMulator prediction"
    )
    prediction_out = state_transformer.inverse_transform(prediction)
    return cast(
        dict[str, jax.Array],
        map_camulator_prediction_arrays(
            constants.earth_radius,
            constants.gravity,
            constants.dry_air_gas_constant,
            constants.water_vapor_mass_ratio_correction,
            constants.dry_air_molecular_weight,
            constants.universal_gas_constant,
            constants.reference_pressure,
            constants.dry_air_kappa,
            constants.stefan_boltzmann_constant,
            camulator_reference_pressure,
            runtime_array_to_host(hyai.cpu().numpy()).squeeze(),
            runtime_array_to_host(hybi.cpu().numpy()).squeeze(),
            runtime_array_to_host(hyam.cpu().numpy()).squeeze(),
            runtime_array_to_host(hybm.cpu().numpy()).squeeze(),
            _camulator_output_array(accessor_output, prediction_out, "U"),
            _camulator_output_array(accessor_output, prediction_out, "V"),
            _camulator_output_array(accessor_output, prediction_out, "TS"),
            _camulator_output_array(accessor_output, prediction_out, "T"),
            _camulator_output_array(
                accessor_output, prediction_out, water_vapor_variable
            ),
            _camulator_output_array(accessor_output, prediction_out, "FSNS"),
            _camulator_output_array(accessor_output, prediction_out, "FLNS"),
            _camulator_output_array(accessor_output, prediction_out, "PS"),
            native_step_seconds=native_step_seconds,
        ),
    )


def map_camulator_initial_runtime_fields(
    resources: CAMulatorGCMSetupState,
    model_state: torch.Tensor,
    cursor: CamulatorRuntimeCursor,
    constants: PhysicalConstants,
) -> dict[str, jax.Array]:
    """Decode the latest native input frame without advancing model or cursor.

    TS/FSNS/FLNS absent from the input must be provided at the exact effective
    cursor time; radiation uses the same native-step energy units as prediction.
    Layout/capability validation precedes inverse transform and any model work.
    """
    import numpy as np

    owner = f"CAMulator {resources.grid.name}"
    vapor = _require_camulator_vapor_channel(
        resources.accessor_input, resources.water_vapor_variable, owner=owner
    )
    _require_camulator_vapor_channel(resources.accessor_output, vapor, owner=owner)
    diagnostics = resources.initial_diagnostics
    if diagnostics is None or "time" not in diagnostics:
        raise ComponentError(f"{owner} initial diagnostics require time.")
    try:
        timestamp = np.asarray(diagnostics["time"])
        if timestamp.shape != ():
            raise ValueError("expected a scalar time")
        time_matches = np.datetime64(cast(Any, timestamp[()])) == np.datetime64(
            cursor.init_datetime
        )
    except (TypeError, ValueError) as exc:
        raise ComponentError(
            f"{owner} initial diagnostics require a scalar timestamp."
        ) from exc
    if not time_matches:
        raise ComponentError(
            f"{owner} initial diagnostics time must match effective initial time {cursor.init_datetime}."
        )
    native_diagnostics: dict[str, object] = {}
    for name in ("TS", "FSNS", "FLNS"):
        try:
            index = resources.accessor_input.get_var_index(name)
        except ValueError:
            index = None
        if index is not None and index.available:
            continue
        if name not in diagnostics:
            raise ComponentError(f"{owner} initial diagnostics require {name}.")
        value = np.asarray(diagnostics[name])
        if value.shape != resources.grid.shape or not np.all(np.isfinite(value)):
            raise ComponentError(
                f"{owner} initial {name} must have finite grid shape {resources.grid.shape}."
            )
        native_diagnostics[name] = value
    if (
        model_state.ndim != 5
        or model_state.shape[0] != 1
        or tuple(model_state.shape[-2:]) != resources.grid.shape
        or model_state.shape[2] < 1
    ):
        raise ComponentError(
            f"{owner} initial input must have shape (1, channel, history, lat, lon)."
        )
    # CREDIT exposes a distinct input inverse; select history before decoding.
    decoded = resources.state_transformer.inverse_transform_input(
        model_state[:, :, -1].clone()
    ).unsqueeze(2)

    def get(name: str) -> object:
        if name in native_diagnostics:
            return native_diagnostics[name]
        try:
            value = resources.accessor_input.get_state_var(decoded, name, time_idx=-1)
        except (ValueError, IndexError) as exc:
            raise ComponentError(f"{owner} initial {name}: {exc}") from exc
        host = runtime_array_to_host(value.detach().cpu().numpy())
        index = resources.accessor_input.get_var_index(name)
        expected = (1, index.n_channels, *resources.grid.shape)
        if host.shape != expected:
            raise ComponentError(
                f"{owner} initial {name} requires shape {expected}; got {host.shape}."
            )
        if not np.all(np.isfinite(host)):
            raise ComponentError(
                f"{owner} initial {name} contains non-finite active values."
            )
        return host

    shape = (resources.conf["model"]["levels"], *resources.grid.shape)
    air = _map_camulator_air_fields(
        constants,
        reference_pressure=resources.P0,
        hyai=resources.hyai.cpu().numpy(),
        hybi=resources.hybi.cpu().numpy(),
        hyam=resources.hyam.cpu().numpy(),
        hybm=resources.hybm.cpu().numpy(),
        u_wind=np.asarray(get("U")).reshape(shape),
        v_wind=np.asarray(get("V")).reshape(shape),
        temperature_3d=np.asarray(get("T")).reshape(shape),
        water_vapor_3d=np.asarray(get(vapor)).reshape(shape),
        surface_pressure=np.asarray(get("PS")).reshape(resources.grid.shape),
    )
    ts = as_jax_real_array(get("TS")).reshape(resources.grid.shape)
    duration = resources.model_timestep.total_seconds()
    return {
        **air,
        "total_surface_temperature": ts,
        "net_shortwave_radiation_flux": as_jax_real_array(get("FSNS")).reshape(
            resources.grid.shape
        )
        / duration,
        "downward_longwave_radiation_flux": constants.stefan_boltzmann_constant * ts**4
        + as_jax_real_array(get("FLNS")).reshape(resources.grid.shape) / duration,
    }


__all__ = [
    "map_camulator_initial_runtime_fields",
    "map_camulator_prediction_arrays",
    "map_camulator_prediction_to_runtime_fields",
    "prepare_camulator_dynamic_forcing_chunk",
    "prepare_camulator_sst_input",
    "prepare_camulator_surface_forcing",
]
