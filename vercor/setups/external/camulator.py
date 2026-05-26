"""CAMulator host-runtime adapter and JAX-backed exchange-field helpers."""

from collections.abc import Mapping
from typing import Any, Optional
from pathlib import Path

import jax.numpy as jnp

from vercor.jax_logging import LoggerLike, get_default_logger

from vercor.setups._time_helpers import (
    assign_model_timestep_alignment,
    seed_grid_field_defaults,
)
import vercor.setups.external.camulator_fields as _camulator_fields
from vercor.setups.external.camulator_forcing import CamulatorRuntimeCursor
import vercor.setups.external.camulator_init as _camulator_init
import vercor.setups.external.camulator_output as _camulator_output
from vercor.setups.external.camulator_runtime_settings import (
    configure_camulator_runtime,
)
import vercor.setups.external.camulator_tensors as _camulator_tensors

from datetime import datetime, timedelta
import xarray as xr

# ---------- #
import torch

from vercor.components import (
    ComponentStepContext,
    HostRuntimeComponent,
    host_component,
)
from vercor.dtypes import jax_ones
from vercor.grid import RectilinearGrid
from vercor.components import ComponentSetupContext
from vercor.types import RuntimeArray

configure_camulator_runtime()

# ============================================================================
# INTEGRATION LOOP
# ============================================================================


class _CAMulatorGCMState:
    coupling_timestep: timedelta
    model_timestep: timedelta
    model_substeps: int

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
        self.runtime_cursor = CamulatorRuntimeCursor()

        context = _camulator_init.initialize_camulator(
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
        context: ComponentSetupContext,
    ) -> None:
        logger = context.logger
        self.coupler_start_datetime = context.start
        assign_model_timestep_alignment(
            self,
            context.dt_seconds,
            timedelta(hours=self.lead_time_periods),
        )
        self.spinup_steps = int(
            self.spinup_time.total_seconds() // self.coupling_timestep.total_seconds()
        )

        # Add noise to initial conditions if requested
        if self.init_noise is not None:
            self.state = _camulator_init.add_init_noise(
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
        self.runtime_cursor.initialize(
            conf=self.conf,
            dynamic_ds=self.dynamic_ds,
            coupler_start_datetime=self.coupler_start_datetime,
            model_substeps=self.model_substeps,
            logger=logger,
        )

        self.accessor_state = _camulator_tensors.StateVariableAccessor(
            self.conf, tensor_type="state"
        )
        self.accessor_input = _camulator_tensors.StateVariableAccessor(
            self.conf, tensor_type="input"
        )
        self.accessor_output = _camulator_tensors.StateVariableAccessor(
            self.conf, tensor_type="output"
        )

        self.forecast_hour = 1
        seed_grid_field_defaults(
            component,
            _camulator_fields._CAMULATOR_RUNTIME_FIELD_NAMES,
            context,
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

        block_start = self.runtime_cursor.current_index()
        block_end = block_start + self.model_substeps

        # Load chunk of dynamic forcing data
        ds_slice = self.dynamic_ds.isel(time=slice(block_start, block_end)).load()
        ds_slice_times = ds_slice["time"].values

        dynamic_forcing_chunk = (
            _camulator_fields._prepare_camulator_dynamic_forcing_chunk(
                ds_slice.to_array(dim="dynamic_variable").values
            )
        )
        gpu_forcing_chunk = _camulator_tensors._torch_tensor_from_jax_array(
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
                    "    CAMulator step: "
                    f"{self.forecast_hour:05}, time: {utc_datetime}"
                )

            dynamic_forcing_t = gpu_forcing_chunk[t].unsqueeze(0)

            # CAMulator's first state already contains forcing. Later states need
            # the next forcing slice appended before inference.
            if self.forecast_hour != 1:
                # Build forcing from dynamic + static
                model_input = self.stepper.state_manager.build_input_with_forcing(
                    self.state, dynamic_forcing_t, self.static_forcing
                )
            else:
                # First iteration: initial state already contains forcing
                model_input = self.state

            total_ts, rescaled_total_ts = (
                _camulator_fields._prepare_camulator_surface_forcing(
                    fields["sea_surface_temperature"],
                    fields["land_surface_temperature"],
                    self.LANDM_COSLAT,
                )
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
                _camulator_tensors._torch_tensor_from_jax_array(
                    _camulator_fields._prepare_camulator_sst_input(rescaled_total_ts),
                    self.device,
                ),
            )

            # Run model
            with torch.no_grad():
                prediction = self.stepper.model(model_input.float())

            # Apply post-processing
            prediction = self.stepper._apply_postprocessing(prediction, model_input)

            _camulator_output.write_camulator_prediction_output(
                prediction,
                utc_datetime,
                latitude=self.latlons.latitude.values,
                longitude=self.latlons.longitude.values,
                init_str=self.runtime_cursor.init_str,
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

            self.forecast_hour += 1

        # ================================================================
        # Deposit final prediction into data dict for coupling (after chunk loop)
        # ================================================================

        if prediction is None or last_total_surface_temperature is None:
            raise ValueError(
                "No CAMulator timesteps were generated from the forcing slice; "
                "check forcing availability and coupling timestep alignment."
            )

        self.runtime_cursor.advance()

        mapped_fields = _camulator_fields._map_camulator_prediction_to_runtime_fields(
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
        outputs=_camulator_fields._CAMULATOR_RUNTIME_FIELD_NAMES,
        default_fields={
            field_name: 0.0
            for field_name in _camulator_fields._CAMULATOR_RUNTIME_FIELD_NAMES
        },
        initialize=state.initialize,
    )


__all__ = [
    "_CAMulatorGCMState",
    "make_camulator_gcm",
]
