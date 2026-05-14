"""CAMulator host-runtime adapter and JAX-backed exchange-field helpers."""

import os

from collections.abc import Mapping
from typing import Any, Optional, cast
from pathlib import Path

import jax
import jax.numpy as jnp

from vercor.jax_logging import LoggerLike, get_default_logger

from setups.external.camulator_state import (
    initialize_camulator,
    parse_datetime_from_config,
    StateVariableAccessor,
)

from datetime import datetime, timedelta
import xarray as xr

# ---------- #
import torch

from vercor.components.base import (
    ComponentStepContext,
    HostRuntimeComponent,
    host_component,
)
from vercor.dtypes import PrecisionPolicy, as_jax_real_array, jax_full, jax_ones
from vercor.fluxes.utilities import _compute_hybrid_sigma_full_level_altitudes
from vercor.grid import RectilinearGrid
from vercor.host_arrays import runtime_array_to_host
from vercor.runtime.contexts import ComponentInitContext
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

make_xarray: Any | None = None
save_netcdf_increment: Any | None = None


def _credit_output_functions() -> tuple[Any, Any]:
    """Load CREDIT output helpers when CAMulator writes a forecast increment."""

    global make_xarray, save_netcdf_increment

    if make_xarray is not None and save_netcdf_increment is not None:
        return make_xarray, save_netcdf_increment

    try:
        from credit.output import (  # type: ignore[import-not-found]
            make_xarray as loaded_make_xarray,
            save_netcdf_increment as loaded_save_netcdf_increment,
        )
    except ModuleNotFoundError as error:
        raise ImportError(
            "CREDIT output helpers are required to write CAMulator forecasts. "
            "Please install credit to use CAMulator output."
        ) from error

    make_xarray = loaded_make_xarray
    save_netcdf_increment = loaded_save_netcdf_increment
    return make_xarray, save_netcdf_increment


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


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


def _torch_tensor_from_jax_array(
    array: RuntimeArray,
    device: str,
    *,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Transfer a JAX-compatible array through an explicit host-to-Torch boundary."""

    tensor = torch.as_tensor(runtime_array_to_host(array).copy())
    if pin_memory and device != "cpu" and torch.cuda.is_available():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=True)


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
    altitude = _compute_hybrid_sigma_full_level_altitudes(
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


def _write_camulator_prediction_output(
    prediction: torch.Tensor,
    utc_datetime: datetime,
    *,
    latitude: object,
    longitude: object,
    init_str: str,
    lead_time_periods: int,
    forecast_hour: int,
    metadata: dict[str, Any],
    conf: dict[str, Any],
) -> None:
    """Write one CAMulator prediction increment through the CREDIT output boundary."""

    credit_make_xarray, credit_save_netcdf_increment = _credit_output_functions()
    upper_air, single_level = credit_make_xarray(
        prediction.cpu(),
        utc_datetime,
        latitude,
        longitude,
        conf,
    )
    credit_save_netcdf_increment(
        upper_air,
        single_level,
        init_str,
        lead_time_periods * forecast_hour,
        metadata,
        conf,
    )


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


def add_init_noise(
    state: torch.Tensor,
    noise_std: float = 0.05,
    logger: LoggerLike | None = None,
) -> torch.Tensor:
    """
    Add random noise to initial conditions for ensemble generation.

    Args:
        state: Initial state tensor
        noise_std: Standard deviation of Gaussian noise

    Returns:
        state_with_noise: Perturbed state
    """
    log = logger if logger is not None else get_default_logger()
    log.info(f"Adding initial condition noise (std={noise_std})")
    noise = torch.randn_like(state) * noise_std
    return state + noise


# ============================================================================
# INTEGRATION LOOP
# ============================================================================


class _CAMulatorGCMState:

    def __init__(
        self,
        config_path: str,
        name: str = "ATM",
        model_weights_path: str = "checkpoint.pt00091.pt",
        output_subfolder_name: Optional[str] = None,
        init_noise: Optional[float] = None,
        spinup_time: timedelta = timedelta(days=2),
        do_spinup: bool = False,
        device: str = "cuda",
        output_cpus_number: int = 8,
        logger: LoggerLike | None = None,
    ) -> None:

        self.logger = logger if logger is not None else get_default_logger()
        self.config_path = config_path
        self.model_weights_path = model_weights_path
        self.device = device
        self.save_append = output_subfolder_name
        self.init_noise = init_noise
        self.output_cpus_number = output_cpus_number
        self.spinup_time = spinup_time
        self.do_spinup = do_spinup

        context = initialize_camulator(
            config_path=self.config_path,
            model_name=self.model_weights_path,
            device=self.device,
            logger=self.logger,
        )

        # Unpack context
        self.conf = context["conf"]
        self.stepper = context["stepper"]
        self.forcing_ds_norm = context["forcing_dataset"]
        self.static_forcing = context["static_forcing"]
        self.state = context["initial_state"]
        self.latlons = context["latlons"]
        self.metadata = context["metadata"]
        self.device = context["device"]
        self.state_transformer = context["state_transformer"]

        # Update save location if append specified
        if self.save_append:
            base = self.conf["predict"].get("save_forecast")
            if not base:
                raise KeyError("'save_forecast' missing in config")
            self.conf["predict"]["save_forecast"] = str(
                Path(base).expanduser() / self.save_append
            )
            self.logger.info(
                f"Saving outputs to: {self.conf['predict']['save_forecast']}"
            )

        # Setup for time-stepping
        self.df_vars = self.conf["data"]["dynamic_forcing_variables"]
        # Time step in hours (e.g., 6 for 6-hour steps)
        self.lead_time_periods = self.conf["data"]["lead_time_periods"]

        grid = RectilinearGrid(
            name=name,
            longitude=self.latlons.longitude.values,
            latitude=self.latlons.latitude.values,
            binary_mask=jax_ones(
                (
                    self.latlons.latitude.values.shape[0],
                    self.latlons.longitude.values.shape[0],
                )
            ),
        )

        self.grid = grid

    def initialize(
        self,
        component: HostRuntimeComponent,
        context: ComponentInitContext,
    ) -> None:
        logger = context.logger
        self.coupler_start_datetime = context.start
        self.coupling_timestep = timedelta(seconds=context.dt_seconds)
        self.spinup_steps = int(
            self.spinup_time.total_seconds() // self.coupling_timestep.total_seconds()
        )
        self.model_timestep = timedelta(hours=self.lead_time_periods)
        self.model_substeps = int(
            self.coupling_timestep.total_seconds()
            // self.model_timestep.total_seconds()
        )

        if self.coupling_timestep % self.model_timestep != timedelta(days=0):
            raise ValueError(
                f"model_timestep ({self.model_timestep}) must be a "
                f"multiple of coupling_timestep ({self.coupling_timestep})"
            )

        # Add noise to initial conditions if requested
        if self.init_noise is not None:
            self.state = add_init_noise(
                self.state,
                noise_std=self.init_noise,
                logger=logger,
            )

        # Trace model for performance (optional but recommended)
        logger.info("Tracing model with torch.jit...")
        # IMPORTANT: Initial state already contains forcing for first timestep
        # So we trace with the initial state shape as-is (DO NOT add forcing channels)
        dummy_input = torch.zeros_like(self.state)
        traced_model = torch.jit.trace(self.stepper.model, dummy_input)
        self.stepper.model = traced_model
        logger.info(f"Model traced with input shape: {dummy_input.shape}")

        ds_physics = xr.open_dataset(self.conf["data"]["save_loc_physics"])

        self.P0 = 100000.0
        self.hyai = torch.tensor(ds_physics["hyai"].values / self.P0).to(self.device)[
            None, :, None, None
        ]
        self.hyam = torch.tensor(ds_physics["hyam"].values).to(self.device)[
            None, :, None, None
        ]
        self.hybi = torch.tensor(ds_physics["hybi"].values).to(self.device)[
            None, :, None, None
        ]
        self.hybm = torch.tensor(ds_physics["hybm"].values).to(self.device)[
            None, :, None, None
        ]
        self.LANDM_COSLAT = ds_physics["LANDM_COSLAT"].values

        # Get forcing data subset
        self.dynamic_ds = self.forcing_ds_norm[self.df_vars]

        # IMPORTANT: Use the config's datetime object directly for xarray lookup
        # It might be cftime.DatetimeNoLeap, which xarray expects
        start_datetime_raw = self.conf["predict"]["start_datetime"]
        loc = self.dynamic_ds.indexes["time"].get_loc(start_datetime_raw)
        self.start_ix = loc.start if isinstance(loc, slice) else loc
        logger.info(f"Starting integration at time index: {self.start_ix}")

        # Now convert to Python datetime for output formatting (if it's a string or cftime)
        init_dt = parse_datetime_from_config(self.conf)
        self.init_str = init_dt.strftime("%Y-%m-%dT%HZ")

        if self.coupler_start_datetime != init_dt:
            logger.warning(
                f"Coupler start datetime ({self.coupler_start_datetime}) does not match "
                f"CAMulator forcing start datetime ({start_datetime_raw}). "
                f"Using CAMulator start datetime for indexing."
            )

        self.accessor_state = StateVariableAccessor(self.conf, tensor_type="state")
        self.accessor_input = StateVariableAccessor(self.conf, tensor_type="input")
        self.accessor_output = StateVariableAccessor(self.conf, tensor_type="output")

        self.forecast_hour = 1
        self.timestep_counter = 0

        component.seed_fields(
            component.grid_field_defaults(
                _CAMULATOR_RUNTIME_FIELD_NAMES,
                policy=context.settings,
            )
        )

    def step(
        self,
        fields: Mapping[str, Any],
        context: ComponentStepContext,
        payload: Any | None,
    ) -> Mapping[str, Any]:
        """Advance the private host-backed CAMulator atmosphere boundary."""

        _ = payload
        time = context.time
        logger = context.logger
        if time is None:
            return {}

        settings = context.settings
        prediction = None
        last_total_surface_temperature: RuntimeArray | None = None

        block_start = self.start_ix + self.timestep_counter * self.model_substeps
        block_end = block_start + self.model_substeps

        # Load chunk of dynamic forcing data
        ds_slice = self.dynamic_ds.isel(time=slice(block_start, block_end)).load()
        ds_slice_times = ds_slice["time"].values

        dynamic_forcing_chunk = _prepare_camulator_dynamic_forcing_chunk(
            ds_slice.to_array(dim="dynamic_variable").values
        )
        gpu_forcing_chunk = _torch_tensor_from_jax_array(
            dynamic_forcing_chunk[:, :, jnp.newaxis, :, :],
            self.device,
            pin_memory=True,
        )

        # Step through each time in the chunk
        for t in range(gpu_forcing_chunk.shape[0]):
            time_obj = ds_slice_times[t]

            # Convert to Python datetime for output formatting.
            # Normalize NumPy scalar time objects before type checks.
            if hasattr(time_obj, "item"):
                time_obj = time_obj.item()

            if isinstance(time_obj, datetime):
                utc_datetime = time_obj
            else:
                # cftime object - convert to Python datetime
                utc_datetime = datetime(
                    time_obj.year,
                    time_obj.month,
                    time_obj.day,
                    time_obj.hour,
                    time_obj.minute,
                    time_obj.second,
                )

            if logger is not None:
                logger.info(
                    f"    CAMulator step: {self.timestep_counter + 1:05}, time: {utc_datetime}"
                )

            dynamic_forcing_t = gpu_forcing_chunk[t].unsqueeze(0)

            # CAMulator's first state already contains forcing. Later states need
            # the next forcing slice appended before inference.
            if self.timestep_counter != 0:
                # Build forcing from dynamic + static
                model_input = self.stepper.state_manager.build_input_with_forcing(
                    self.state, dynamic_forcing_t, self.static_forcing
                )
            else:
                # First iteration: initial state already contains forcing
                model_input = self.state

            total_ts, rescaled_total_ts = _prepare_camulator_surface_forcing(
                fields["sea_surface_temperature"],
                fields["land_surface_temperature"],
                self.LANDM_COSLAT,
            )
            last_total_surface_temperature = total_ts

            # Land surface temperature is already rescaled in the same way as sst
            if logger is not None:
                logger.info(
                    "    Rescaled ts stats - max: "
                    f"{float(jnp.max(rescaled_total_ts)):.4f}, min: "
                    f"{float(jnp.min(rescaled_total_ts)):.4f}"
                )

            self.accessor_input.set_state_var(
                model_input,
                "SST",
                _torch_tensor_from_jax_array(
                    _prepare_camulator_sst_input(rescaled_total_ts), self.device
                ),
            )

            # Run model
            with torch.no_grad():
                prediction = self.stepper.model(model_input.float())

            # Apply post-processing
            prediction = self.stepper._apply_postprocessing(prediction, model_input)

            _write_camulator_prediction_output(
                prediction,
                utc_datetime,
                latitude=self.latlons.latitude.values,
                longitude=self.latlons.longitude.values,
                init_str=self.init_str,
                lead_time_periods=self.lead_time_periods,
                forecast_hour=self.forecast_hour,
                metadata=self.metadata,
                conf=self.conf,
            )

            # ================================================================
            # SHIFT STATE FORWARD FOR NEXT TIMESTEP
            # ================================================================

            self.state = self.stepper.state_manager.shift_state_forward(
                self.state, prediction
            )

            self.timestep_counter += 1
            self.forecast_hour += 1

        # ================================================================
        # Deposit final prediction into data dict for coupling (after chunk loop)
        # ================================================================

        if prediction is None or last_total_surface_temperature is None:
            raise ValueError(
                "No CAMulator timesteps were generated from the forcing slice; "
                "check forcing availability and coupling timestep alignment."
            )

        mapped_fields = _map_camulator_prediction_to_runtime_fields(
            settings,
            camulator_reference_pressure=self.P0,
            hyai=self.hyai,
            hybi=self.hybi,
            hyam=self.hyam,
            hybm=self.hybm,
            accessor_output=self.accessor_output,
            state_transformer=self.state_transformer,
            prediction=prediction,
        )

        return {
            "total_surface_temperature": last_total_surface_temperature,
            **mapped_fields,
        }

    def step_host_runtime_state(
        self,
        component_state: Any,
        context: ComponentStepContext,
    ) -> Any:
        """Compatibility helper for state-level unit tests."""

        updates = self.step(component_state.data.to_mapping(), context, None)
        return component_state.with_data(component_state.data.set_many(updates))


def make_camulator_gcm(
    config_path: str,
    name: str = "ATM",
    model_weights_path: str = "checkpoint.pt00091.pt",
    output_subfolder_name: Optional[str] = None,
    init_noise: Optional[float] = None,
    spinup_time: timedelta = timedelta(days=2),
    do_spinup: bool = False,
    device: str = "cuda",
    output_cpus_number: int = 8,
    logger: LoggerLike | None = None,
) -> HostRuntimeComponent:
    """Return a host-backed CAMulator atmosphere component."""

    state = _CAMulatorGCMState(
        config_path=config_path,
        name=name,
        model_weights_path=model_weights_path,
        output_subfolder_name=output_subfolder_name,
        init_noise=init_noise,
        spinup_time=spinup_time,
        do_spinup=do_spinup,
        device=device,
        output_cpus_number=output_cpus_number,
        logger=logger,
    )
    return host_component(
        name=name,
        grid=state.grid,
        step=state.step,
        inputs=("sea_surface_temperature", "land_surface_temperature"),
        outputs=_CAMULATOR_RUNTIME_FIELD_NAMES,
        default_fields={
            field_name: 0.0 for field_name in _CAMULATOR_RUNTIME_FIELD_NAMES
        },
        initialize=state.initialize,
    )
