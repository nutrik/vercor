"""
Quick_Climate_V02.py
--------------------
Refactored CAMulator climate integration with clearer coupling interfaces.

Key improvements:
- Separated initialization from time-stepping
- Clear CAMulatorStepper class for coupling
- Documented state tensor structure
- Removed dead code
- Preserved async parallel I/O for performance
"""

import os

from typing import TYPE_CHECKING, Optional
from pathlib import Path

from vercor.components.external.camulator_state import (
    initialize_camulator,
    StateVariableAccessor,
)
from vercor.fluxes.utilities import (
    compute_air_density,
    compute_potential_temperature,
    get_altitudes_hybrid_sigma_levels,
)

from datetime import datetime, timedelta
import numpy as np
import xarray as xr

# ---------- #
import torch

# ---------- #
# credit
try:
    from credit.output import make_xarray, save_netcdf_increment
except ModuleNotFoundError:
    print(
        "Credit module not found. Please install credit to enable NetCDF output functionality."
    )

from vercor.clock import ModelDateTime
from vercor.components.base import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def add_init_noise(state: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    Add random noise to initial conditions for ensemble generation.

    Args:
        state: Initial state tensor
        noise_std: Standard deviation of Gaussian noise

    Returns:
        state_with_noise: Perturbed state
    """
    print(f"Adding initial condition noise (std={noise_std})")
    noise = torch.randn_like(state) * noise_std
    return state + noise


def parse_datetime_from_config(conf: dict) -> datetime:
    """
    Parse datetime from config, handling string, datetime, and cftime objects.

    Args:
        conf: Configuration dictionary

    Returns:
        init_dt: Python datetime object
    """
    raw_dt = conf["predict"]["start_datetime"]

    if isinstance(raw_dt, str):
        # Parse "YYYY-MM-DD HH:MM:SS" format
        return datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S")
    elif isinstance(raw_dt, datetime):
        # Already a Python datetime
        return raw_dt
    else:
        # Assume it's a cftime object - convert to Python datetime
        # cftime objects have year, month, day, hour, minute, second attributes
        return datetime(
            raw_dt.year,
            raw_dt.month,
            raw_dt.day,
            raw_dt.hour,
            raw_dt.minute,
            raw_dt.second,
        )


# ============================================================================
# INTEGRATION LOOP
# ============================================================================


class CAMulatorGCM(Component):

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
    ) -> None:

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
            print(f"Saving outputs to: {self.conf['predict']['save_forecast']}")

        # Setup for time-stepping
        self.df_vars = self.conf["data"]["dynamic_forcing_variables"]
        # Total number of CAMulator steps to run (e.g., 40 for 10-day forecast with 6-hour steps)
        self.num_ts = self.conf["predict"]["timesteps_fast_climate"]
        # Time step in hours (e.g., 6 for 6-hour steps)
        self.lead_time_periods = self.conf["data"]["lead_time_periods"]
        # Maximum ???
        self.chunk_size = self.conf["data"].get("forcing_chunk_size", 32)
        # post_conf = self.conf["model"]["post_conf"]
        # lon_lat_level_names = post_conf["global_mass_fixer"]["lon_lat_level_name"]

        grid = RectilinearGrid(
            name=name,
            longitude=self.latlons.longitude.values,
            latitude=self.latlons.latitude.values,
            binary_mask=np.ones(
                (
                    self.latlons.latitude.values.shape[0],
                    self.latlons.longitude.values.shape[0],
                ),
            ),
        )

        super().__init__(name, grid=grid)

    def initialize(self, coupler: "Coupler") -> None:
        logger = coupler.logger
        self.coupler_start_datetime = coupler.clock.start
        self.coupling_timestep = timedelta(seconds=coupler.clock.dt_seconds)
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
            self.state = add_init_noise(self.state, noise_std=self.init_noise)

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

        zeros = np.zeros(self.grid.shape)
        self.data["specific_humidity"] = zeros.copy()
        self.data["net_shortwave_radiation_flux"] = zeros.copy()
        self.data["downward_longwave_radiation_flux"] = zeros.copy()
        self.data["sea_surface_temperature"] = zeros.copy()
        self.data["land_surface_temperature"] = zeros.copy()
        self.data["u_velocity"] = zeros.copy()
        self.data["v_velocity"] = zeros.copy()
        self.data["temperature"] = zeros.copy()
        self.data["potential_temperature"] = zeros.copy()
        self.data["density"] = zeros.copy()
        self.data["latent_heat_flux"] = zeros.copy()
        self.data["sensible_heat_flux"] = zeros.copy()
        self.data["model_level_height"] = zeros.copy()

    def step(
        self,
        dt: timedelta,
        time: datetime | ModelDateTime,
        coupler: "Coupler",
    ) -> None:

        settings = coupler.settings
        logger = coupler.logger
        data = self.data

        prediction = None

        # block_end = min(block_start + self.chunk_size, self.start_ix + self.num_ts)
        block_start = self.start_ix + self.timestep_counter * self.model_substeps
        block_end = block_start + self.model_substeps

        # Load chunk of dynamic forcing data
        ds_slice = self.dynamic_ds.isel(time=slice(block_start, block_end)).load()
        ds_slice_times = ds_slice["time"].values

        # Stack forcing variables into tensor [time, vars, lat, lon]
        arr_list = [ds_slice[var].values for var in self.dynamic_ds.data_vars]
        arr = np.stack(arr_list, axis=1)

        # Transfer to GPU once per chunk
        cpu_tensor = torch.from_numpy(arr).unsqueeze(2).pin_memory()
        gpu_forcing_chunk = cpu_tensor.to(self.device, non_blocking=True)

        # Step through each time in the chunk
        for t in range(gpu_forcing_chunk.shape[0]):
            time_obj = ds_slice_times[t]

            # Convert to Python datetime for output formatting
            # Handle numpy scalar wrapper
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

            logger.info(
                f"    CAMulator step: {self.timestep_counter + 1:05}, time: {utc_datetime}"
            )

            dynamic_forcing_t = gpu_forcing_chunk[t].unsqueeze(0)

            # ================================================================
            # CORE PHYSICS STEP
            # This matches the original Quick_Climate.py logic exactly:
            # - First step (timestep_counter=0): state already has forcing, run model as-is
            # - Subsequent steps: add forcing to state, then run model
            # ================================================================

            if self.timestep_counter != 0:
                # Build forcing from dynamic + static
                model_input = self.stepper.state_manager.build_input_with_forcing(
                    self.state, dynamic_forcing_t, self.static_forcing
                )
            else:
                # First iteration: initial state already contains forcing
                model_input = self.state

            # once the coupler has run, set the variable: NOTE this needs to be rescaled for our ML model.
            # !!! NOTE this needs to be rescaled for our ML model. !!!
            sst = self.data["sea_surface_temperature"]
            rescaled_sst = (sst - np.nanmean(sst)) / np.nanstd(sst)
            rescaled_sst = np.nan_to_num(rescaled_sst, nan=283.0)

            self.accessor_input.set_state_var(
                model_input,
                "SST",
                torch.tensor(rescaled_sst[np.newaxis, np.newaxis, np.newaxis, ...]).to(
                    self.device
                ),
            )

            # Run model
            with torch.no_grad():
                prediction = self.stepper.model(model_input.float())

            # Apply post-processing
            prediction = self.stepper._apply_postprocessing(prediction, model_input)

            # ================================================================
            # OUTPUT GENERATION (runs in parallel via multiprocessing)
            # ================================================================

            # Convert prediction to xarray (fast, on CPU)
            upper_air, single_level = make_xarray(
                prediction.cpu(),
                utc_datetime,
                self.latlons.latitude.values,
                self.latlons.longitude.values,
                self.conf,
            )

            # save to NetCDF (runs in background pool)
            save_netcdf_increment(
                upper_air,
                single_level,
                self.init_str,
                self.lead_time_periods * self.forecast_hour,
                self.metadata,
                self.conf,
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

        if prediction is None:
            raise ValueError(
                "No CAMulator timesteps were generated from the forcing slice; "
                "check forcing availability and coupling timestep alignment."
            )

        # get all of the variables for coupling:
        prediction_out = self.state_transformer.inverse_transform(prediction)

        # Units: [m/s]
        data["u_velocity"] = np.asarray(
            self.accessor_output.get_state_var(prediction_out, "U")
            .cpu()
            .squeeze()[-1, :, :]
        )
        # Units: [m/s]
        data["v_velocity"] = np.asarray(
            self.accessor_output.get_state_var(prediction_out, "V")
            .cpu()
            .squeeze()[-1, :, :]
        )
        # Units: [K]
        TS = np.asarray(
            self.accessor_output.get_state_var(prediction_out, "TS").cpu()
        )  # surface temp
        # Units: [K]
        data["temperature_3d"] = np.asarray(
            self.accessor_output.get_state_var(prediction_out, "T").cpu().squeeze()
        )  # temperature
        # Units: [kg/kg]
        data["specific_humidity_3d"] = np.asarray(
            self.accessor_output.get_state_var(prediction_out, "Qtot").cpu().squeeze()
        )  # specific humidty
        # Near surface temperature
        data["temperature"] = data["temperature_3d"][-1, ...]
        FSNS = self.accessor_output.get_state_var(prediction_out, "FSNS")
        # average radiative flux during 6-hour period in [J/m²] convert to [W/m²]
        # 6 × 3600 = 21600
        FSNS /= 21600
        data["net_shortwave_radiation_flux"] = np.asarray(FSNS.cpu().squeeze())

        FLNS = np.asarray(
            self.accessor_output.get_state_var(prediction_out, "FLNS").cpu()
        )  # FLDS≈εσTs{^4}−FLNS  # will have to approximate it. where emissivity in CAM = 1
        FLNS /= -21600  # J/m² back in CAM units [W/m2]
        FLDS = settings.stefBoltz * TS[...] ** 4 - FLNS
        # Units: [W/m²]
        data["downward_longwave_radiation_flux"] = np.asarray(FLDS.squeeze())

        # Pressure model levels:
        # Units: [Pa]
        PS = self.accessor_output.get_state_var(
            prediction_out, "PS"
        )  # surface pressure
        Pmid = np.asarray(
            (self.hyam * self.P0 + self.hybm * PS).cpu().squeeze()
        )  # pm(k) = Am(k) P0 + Bm(k) PS
        Pint = np.asarray(
            (self.hyai * self.P0 + self.hybi * PS).cpu().squeeze()
        )  # pi(k) = Ai(k) P0 + Bi(k) PS

        # Units: [m]
        data["model_level_height"] = get_altitudes_hybrid_sigma_levels(
            settings,
            data["temperature_3d"].T,
            data["specific_humidity_3d"].T,
            Pint[...].T,
        )[..., -1].T
        # Units: [kg/m³]
        data["density"] = compute_air_density(
            settings, Pmid[-1, :, :], data["temperature"][:, :]
        )
        # Units: [K]
        data["potential_temperature"] = compute_potential_temperature(
            settings, data["temperature"][:, :], Pmid[-1, :, :]
        )
