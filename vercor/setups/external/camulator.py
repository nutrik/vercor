"""CAMulator host-runtime adapter and JAX-backed exchange-field helpers."""

from collections.abc import Mapping
from typing import Any, Optional
from pathlib import Path

from vercor.jax_logging import LoggerLike, get_default_logger

from vercor.setups._time_helpers import (
    assign_model_timestep_alignment,
    seed_grid_field_defaults,
)
import vercor.setups.external.camulator_contracts as _camulator_contracts
from vercor.setups.external.camulator_forcing import CamulatorRuntimeCursor
import vercor.setups.external.camulator_init as _camulator_init
import vercor.setups.external.camulator_runtime as _camulator_runtime
from vercor.setups.external.camulator_runtime_settings import (
    configure_camulator_runtime,
)
import vercor.setups.external.camulator_tensors as _camulator_tensors

from datetime import timedelta
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

configure_camulator_runtime()


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
            _camulator_contracts.CAMULATOR_RUNTIME_FIELD_NAMES,
            context,
        )

    def step(
        self,
        fields: Mapping[str, Any],
        context: ComponentStepContext,
        payload: Any | None,
    ) -> Mapping[str, Any]:
        """Delegate CAMulator host-runtime advancement to the runtime helper."""

        return _camulator_runtime.step_camulator_runtime(
            self,
            fields,
            context,
            payload,
        )


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
        outputs=_camulator_contracts.CAMULATOR_RUNTIME_FIELD_NAMES,
        default_fields=_camulator_contracts.camulator_runtime_field_defaults(),
        initialize=state.initialize,
    )


__all__ = [
    "make_camulator_gcm",
]
