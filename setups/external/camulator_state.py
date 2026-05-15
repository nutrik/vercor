"""
State management and variable access for CAMulator climate simulations.

This module provides modular components for:
- Variable indexing and access in state tensors
- State transformation and time-stepping
- Core CAMulator integration interface with full physics post-processing

Key Classes:
- StateVariableAccessor: Get/set variables by name from state tensors
- StateManager: Handle state transformations and time-stepping
- CAMulatorStepper: Core time-stepping interface with conservation fixers
"""

import os
import yaml
import xarray as xr
import torch
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Literal, Sequence

from setups._time_helpers import runtime_forcing_index
from vercor.jax_logging import LoggerLike, get_default_logger

CREDIT_AVAILABLE = False
POSTBLOCK_AVAILABLE = False
WINDPP_AVAILABLE = False

load_model: Any = None
load_model_name: Any = None
load_transforms: Any = None
Normalize_ERA5_and_Forcing: Any = None
distributed_model_wrapper: Any = None
load_model_state: Any = None
credit_main_parser: Any = None
load_metadata: Any = None

GlobalMassFixer: Any = None
GlobalWaterFixer: Any = None
GlobalEnergyFixer: Any = None
post_process_wind_artifacts: Any = None

logger = get_default_logger()


def _load_credit_modules() -> None:
    """Load CREDIT core modules at the CAMulator execution boundary."""

    global CREDIT_AVAILABLE
    global load_model, load_model_name, load_transforms, Normalize_ERA5_and_Forcing
    global distributed_model_wrapper, load_model_state, credit_main_parser
    global load_metadata

    if CREDIT_AVAILABLE:
        return

    try:
        from credit.distributed import (  # type: ignore[import-not-found]
            distributed_model_wrapper as credit_distributed_model_wrapper,
        )
        from credit.models import (  # type: ignore[import-not-found]
            load_model as credit_load_model,
        )
        from credit.models import (  # type: ignore[import-not-found]
            load_model_name as credit_load_model_name,
        )
        from credit.models.checkpoint import (  # type: ignore[import-not-found]
            load_model_state as credit_load_model_state,
        )
        from credit.output import (  # type: ignore[import-not-found]
            load_metadata as credit_load_metadata,
        )
        from credit.parser import (  # type: ignore[import-not-found]
            credit_main_parser as credit_credit_main_parser,
        )
        from credit.transforms import (  # type: ignore[import-not-found]
            Normalize_ERA5_and_Forcing as credit_normalize,
        )
        from credit.transforms import (  # type: ignore[import-not-found]
            load_transforms as credit_load_transforms,
        )
    except ImportError as error:
        raise ImportError(
            "CREDIT modules are required to initialize CAMulator. "
            "Please ensure credit is installed and importable."
        ) from error

    load_model = credit_load_model
    load_model_name = credit_load_model_name
    load_transforms = credit_load_transforms
    Normalize_ERA5_and_Forcing = credit_normalize
    distributed_model_wrapper = credit_distributed_model_wrapper
    load_model_state = credit_load_model_state
    credit_main_parser = credit_credit_main_parser
    load_metadata = credit_load_metadata
    CREDIT_AVAILABLE = True


def _load_postblock_modules() -> bool:
    """Load optional CREDIT postblock fixers without import-time warnings."""

    global POSTBLOCK_AVAILABLE
    global GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

    if POSTBLOCK_AVAILABLE:
        return True

    try:
        from credit.postblock import (  # type: ignore[import-not-found]
            GlobalEnergyFixer as credit_energy_fixer,
        )
        from credit.postblock import (  # type: ignore[import-not-found]
            GlobalMassFixer as credit_mass_fixer,
        )
        from credit.postblock import (  # type: ignore[import-not-found]
            GlobalWaterFixer as credit_water_fixer,
        )
    except ImportError:
        return False

    GlobalMassFixer = credit_mass_fixer
    GlobalWaterFixer = credit_water_fixer
    GlobalEnergyFixer = credit_energy_fixer
    POSTBLOCK_AVAILABLE = True
    return True


def _load_windpp_module() -> bool:
    """Load optional wind post-processing only when CAMulator stepping needs it."""

    global WINDPP_AVAILABLE, post_process_wind_artifacts

    if WINDPP_AVAILABLE:
        return True

    try:
        from setups.external.windpp import (
            post_process_wind_artifacts as windpp_post_process,
        )
    except ImportError:
        return False

    post_process_wind_artifacts = windpp_post_process
    WINDPP_AVAILABLE = True
    return True


def load_camulator_forcing_context(config_path: str) -> dict[str, Any]:
    """Load CAMulator config and raw forcing without constructing the model."""

    _load_credit_modules()

    with open(config_path) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=False, parse_predict=True, print_summary=False
    )
    conf["predict"]["mode"] = None

    forcing_file = conf["predict"]["forcing_file"]
    if not os.path.exists(forcing_file):
        raise FileNotFoundError(f"Forcing file not found: {forcing_file}")

    chunk_size = conf["data"].get("forcing_chunk_size", 32)
    forcing_ds = xr.open_dataset(forcing_file, chunks={"time": chunk_size})
    return {
        "conf": conf,
        "forcing_dataset_raw": forcing_ds.chunk({"time": chunk_size}),
    }


def parse_datetime_from_config(conf: dict[str, Any]) -> datetime:
    """Parse CAMulator start datetime values into Python ``datetime`` objects."""

    raw_dt = conf["predict"]["start_datetime"]

    if isinstance(raw_dt, str):
        return datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S")
    if isinstance(raw_dt, datetime):
        return raw_dt
    return datetime(
        raw_dt.year,
        raw_dt.month,
        raw_dt.day,
        raw_dt.hour,
        raw_dt.minute,
        raw_dt.second,
    )


@dataclass(frozen=True)
class CAMulatorForcingCursor:
    """Time-index cursor for CAMulator forcing datasets."""

    start_ix: int
    init_datetime: datetime
    init_str: str


@dataclass
class CamulatorRuntimeCursor:
    """Mutable CAMulator forcing cursor shared by host setup adapters."""

    start_ix: int = 0
    init_datetime: datetime | None = None
    init_str: str = ""
    model_substeps: int = 0
    timestep_counter: int = 0

    def initialize(
        self,
        *,
        conf: dict[str, Any],
        dynamic_ds: Any,
        coupler_start_datetime: object,
        model_substeps: int,
        logger: Any,
    ) -> CAMulatorForcingCursor:
        """Initialize forcing index metadata and reset the runtime counter."""

        cursor = initialize_camulator_forcing_cursor(
            conf=conf,
            dynamic_ds=dynamic_ds,
            coupler_start_datetime=coupler_start_datetime,
            logger=logger,
        )
        self.start_ix = cursor.start_ix
        self.init_datetime = cursor.init_datetime
        self.init_str = cursor.init_str
        self.model_substeps = int(model_substeps)
        self.timestep_counter = 0
        return cursor

    def current_index(self) -> int:
        """Return the current forcing index for this cursor."""

        return runtime_forcing_index(
            start_ix=self.start_ix,
            timestep_counter=self.timestep_counter,
            model_substeps=self.model_substeps,
        )

    def advance(self) -> None:
        """Advance the cursor by one runtime counter step."""

        self.timestep_counter += 1


def initialize_camulator_forcing_cursor(
    *,
    conf: dict[str, Any],
    dynamic_ds: Any,
    coupler_start_datetime: object,
    logger: Any,
) -> CAMulatorForcingCursor:
    """Initialize CAMulator forcing time indexing from config and xarray indexes."""

    start_datetime_raw = conf["predict"]["start_datetime"]
    loc = dynamic_ds.indexes["time"].get_loc(start_datetime_raw)
    start_ix = loc.start if isinstance(loc, slice) else int(loc)
    logger.info(f"Starting integration at time index: {start_ix}")

    init_datetime = parse_datetime_from_config(conf)
    init_str = init_datetime.strftime("%Y-%m-%dT%HZ")

    if coupler_start_datetime != init_datetime:
        logger.warning(
            f"Coupler start datetime ({coupler_start_datetime}) does not match "
            f"CAMulator forcing start datetime ({start_datetime_raw}). "
            f"Using CAMulator start datetime for indexing."
        )

    return CAMulatorForcingCursor(
        start_ix=start_ix,
        init_datetime=init_datetime,
        init_str=init_str,
    )


def _append_indexed_variables(
    indices: dict[str, dict[str, Any]],
    variable_names: Sequence[str],
    *,
    start_index: int,
    n_channels: int,
    is_3d: bool,
) -> int:
    """Append available tensor-variable indices and return the next channel index."""

    idx = start_index
    for var in variable_names:
        indices[var] = {
            "start_idx": idx,
            "end_idx": idx + n_channels,
            "n_channels": n_channels,
            "is_3d": is_3d,
            "available": True,
        }
        idx += n_channels
    return idx


def _mark_unavailable_variables(
    indices: dict[str, dict[str, Any]],
    variable_names: Sequence[str],
    *,
    reason: str,
) -> None:
    """Mark variables that are recognized by config but absent from a tensor type."""

    for var in variable_names:
        indices[var] = {"available": False, "reason": reason}


def _prepare_static_forcing_tensor(
    forcing_ds: xr.Dataset,
    static_variables: list[str],
    device: Any,
) -> torch.Tensor:
    """Prepare static CAMulator forcing through an explicit xarray/Torch boundary."""

    static_values = forcing_ds[static_variables].to_array(dim="static_variable").values
    return (
        torch.as_tensor(static_values)
        .unsqueeze(0)
        .unsqueeze(2)
        .to(
            device,
            non_blocking=True,
        )
    )


# ============================================================================
# STATE VARIABLE ACCESSOR - GET/SET VARIABLES BY NAME
# ============================================================================


class StateVariableAccessor:
    """
    Access variables by name in CAMulator state tensors.

    Handles the complexity of variable ordering across different tensor types:
    - 'state': Pure atmospheric state (prognostic + surface, no forcing/diagnostics)
    - 'input': Model input (state + forcing variables)
    - 'output': Model prediction (state + diagnostic variables)

    Usage:
        # Create accessor for your tensor type
        accessor = StateVariableAccessor(conf, tensor_type='state')

        # Get a variable (returns view, not copy)
        u_wind = accessor.get_state_var(state_tensor, 'U')
        # Returns: [batch, levels, time, lat, lon] for 3D variables
        #          [batch, 1, time, lat, lon] for 2D variables

        # Set a variable (modifies in-place)
        accessor.set_state_var(state_tensor, 'U', new_u_values)

        # Query variable info
        info = accessor.get_var_info('U')
        # Returns: {'start_idx': 0, 'end_idx': 32, 'n_channels': 32,
        #           'is_3d': True, 'available': True}

    Tensor Types:
        'state': [prognostic_vars * levels + surface_vars]
        'input': [forcing + prognostic_vars * levels + surface_vars]
        'output': [prognostic_vars * levels + surface_vars + diagnostic_vars]
    """

    def __init__(
        self, conf: dict, tensor_type: Literal["state", "input", "output"] = "state"
    ):
        """
        Initialize variable accessor.

        Args:
            conf: Configuration dictionary with data section
            tensor_type: Type of tensor to index into
                'state': Pure state without forcing or diagnostics
                'input': Model input with forcing added
                'output': Model prediction with diagnostics
        """
        self.conf = conf
        self.tensor_type = tensor_type

        # Extract variable lists from config
        self.prognostic_vars = conf["data"]["variables"]  # 3D upper-air
        self.surface_vars = conf["data"]["surface_variables"]  # 2D surface
        self.diagnostic_vars = conf["data"][
            "diagnostic_variables"
        ]  # 2D diagnostics (output only)
        self.dynamic_forcing_vars = conf["data"][
            "dynamic_forcing_variables"
        ]  # 2D time-varying
        self.forcing_vars = conf["data"]["forcing_variables"]  # 2D periodic
        self.static_vars = conf["data"]["static_variables"]  # 2D static

        self.levels = conf["model"]["levels"]
        self.static_first = conf["data"]["static_first"]

        # Build index maps for each tensor type
        self._build_index_maps()

    def _build_index_maps(self) -> None:
        """Build index mappings for all variables in each tensor type."""
        self.var_indices: dict[str, dict] = {}

        # State tensor: [prognostic * levels + surface]
        self._build_state_indices()

        # Input tensor: [forcing + state]
        self._build_input_indices()

        # Output tensor: [prognostic * levels + surface + diagnostics]
        self._build_output_indices()

    def _build_state_indices(self) -> None:
        """Build indices for pure state tensor (no forcing, no diagnostics)."""
        indices: dict[str, dict[str, Any]] = {}
        idx = 0

        idx = _append_indexed_variables(
            indices,
            self.prognostic_vars,
            start_index=idx,
            n_channels=self.levels,
            is_3d=True,
        )
        _append_indexed_variables(
            indices,
            self.surface_vars,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )
        _mark_unavailable_variables(
            indices,
            self.diagnostic_vars,
            reason="Diagnostics not in state tensor",
        )
        _mark_unavailable_variables(
            indices,
            (*self.dynamic_forcing_vars, *self.forcing_vars, *self.static_vars),
            reason="Forcing not in state tensor",
        )

        self.var_indices["state"] = indices

    def _build_input_indices(self) -> None:
        """Build indices for model input tensor (with forcing).

        Input tensor structure: [state] + [forcing]
        Where state = prognostic + surface
        """
        indices: dict[str, dict[str, Any]] = {}
        idx = 0

        idx = _append_indexed_variables(
            indices,
            self.prognostic_vars,
            start_index=idx,
            n_channels=self.levels,
            is_3d=True,
        )
        idx = _append_indexed_variables(
            indices,
            self.surface_vars,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )

        # THIRD: Forcing variables - appended after state
        # Order depends on static_first flag
        if self.static_first:
            forcing_order = (
                self.static_vars + self.dynamic_forcing_vars + self.forcing_vars
            )
        else:
            forcing_order = (
                self.dynamic_forcing_vars + self.forcing_vars + self.static_vars
            )

        _append_indexed_variables(
            indices,
            forcing_order,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )
        _mark_unavailable_variables(
            indices,
            self.diagnostic_vars,
            reason="Diagnostics not in input tensor",
        )

        self.var_indices["input"] = indices

    def _build_output_indices(self) -> None:
        """Build indices for model output tensor (with diagnostics)."""
        indices: dict[str, dict[str, Any]] = {}
        idx = 0

        idx = _append_indexed_variables(
            indices,
            self.prognostic_vars,
            start_index=idx,
            n_channels=self.levels,
            is_3d=True,
        )
        idx = _append_indexed_variables(
            indices,
            self.surface_vars,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )
        _append_indexed_variables(
            indices,
            self.diagnostic_vars,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )
        _mark_unavailable_variables(
            indices,
            (*self.dynamic_forcing_vars, *self.forcing_vars, *self.static_vars),
            reason="Forcing not in output tensor",
        )

        self.var_indices["output"] = indices

    def get_var_info(self, var_name: str) -> Any:
        """
        Get indexing information for a variable.

        Args:
            var_name: Variable name (e.g., 'U', 'TAUX', 'PS')

        Returns:
            Dictionary with variable info:
                - available: Whether variable exists in current tensor type
                - start_idx, end_idx, n_channels, is_3d: If available
                - reason: If not available, why not

        Raises:
            ValueError: If variable name not recognized in config
        """
        indices = self.var_indices[self.tensor_type]

        if var_name not in indices:
            all_vars = (
                self.prognostic_vars
                + self.surface_vars
                + self.diagnostic_vars
                + self.dynamic_forcing_vars
                + self.forcing_vars
                + self.static_vars
            )
            raise ValueError(
                f"Variable '{var_name}' not found in config. Available variables: {all_vars}"
            )

        return indices[var_name]

    def get_state_var(
        self, state_tensor: torch.Tensor, var_name: str, time_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract a variable from the state tensor.

        Args:
            state_tensor: State tensor [batch, channels, time, lat, lon]
            var_name: Variable name (e.g., 'U', 'TAUX', 'PS')
            time_idx: Optional time index to extract. If None, returns all times.

        Returns:
            Variable tensor:
                - 3D vars: [batch, levels, time, lat, lon] or [batch, levels, lat, lon]
                - 2D vars: [batch, 1, time, lat, lon] or [batch, 1, lat, lon]

        Raises:
            ValueError: If variable not available in current tensor type
            IndexError: If time_idx out of bounds
        """
        info = self.get_var_info(var_name)

        if not info["available"]:
            raise ValueError(
                f"Variable '{var_name}' not available in '{self.tensor_type}' tensor. "
                f"Reason: {info.get('reason', 'Unknown')}"
            )

        # Extract variable slice
        var_slice = state_tensor[
            :, info["start_idx"] : info["end_idx"], ...  # noqa: E203
        ]

        # Extract specific time if requested
        if time_idx is not None:
            if time_idx >= state_tensor.shape[2]:
                raise IndexError(
                    f"Time index {time_idx} out of bounds for tensor with {state_tensor.shape[2]} time steps"
                )
            var_slice = var_slice[:, :, time_idx, :, :]

        return var_slice

    def set_state_var(
        self,
        state_tensor: torch.Tensor,
        var_name: str,
        var_data: torch.Tensor,
        time_idx: Optional[int] = None,
    ) -> None:
        """
        Set a variable in the state tensor (in-place modification).

        Args:
            state_tensor: State tensor to modify [batch, channels, time, lat, lon]
            var_name: Variable name (e.g., 'U', 'TAUX', 'PS')
            var_data: New data for variable. Must match expected shape.
            time_idx: Optional time index to set. If None, sets all times.

        Raises:
            ValueError: If variable not available or shape mismatch
        """
        info = self.get_var_info(var_name)

        if not info["available"]:
            raise ValueError(
                f"Variable '{var_name}' not available in '{self.tensor_type}' tensor. "
                f"Reason: {info.get('reason', 'Unknown')}"
            )

        # Validate shape
        expected_shape: tuple[int, ...]
        if time_idx is None:
            expected_shape = (
                state_tensor.shape[0],  # batch
                info["n_channels"],  # channels (levels or 1)
                state_tensor.shape[2],  # time
                state_tensor.shape[3],  # lat
                state_tensor.shape[4],  # lon
            )
        else:
            expected_shape = (
                state_tensor.shape[0],  # batch
                info["n_channels"],  # channels
                state_tensor.shape[3],  # lat
                state_tensor.shape[4],  # lon
            )

        if var_data.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for '{var_name}'. Expected {expected_shape}, got {var_data.shape}"
            )

        # Set variable (in-place)
        if time_idx is None:
            state_tensor[:, info["start_idx"] : info["end_idx"], ...] = (  # noqa: E203
                var_data
            )
        else:
            state_tensor[
                :, info["start_idx"] : info["end_idx"], time_idx, :, :  # noqa: E203
            ] = var_data

    def list_available_vars(self) -> dict[str, dict]:
        """
        List all variables available in current tensor type.

        Returns:
            Dictionary mapping variable names to their info dicts
        """
        return {
            var: info
            for var, info in self.var_indices[self.tensor_type].items()
            if info.get("available", False)
        }


# ============================================================================
# STATE MANAGER - TRANSFORMATIONS AND TIME-STEPPING
# ============================================================================


class StateManager:
    """
    Manages the CAMulator state tensor structure and transformations.

    State Tensor Structure
    ----------------------
    The state tensor contains atmospheric variables over multiple timesteps (history).

    Dimensions: [batch, channels, time, lat, lon]

    IMPORTANT FOR COUPLING:
    -----------------------
    The INITIAL state loaded from file already includes forcing for the first timestep.
    After the first step, shift_state_forward() returns atmospheric state WITHOUT forcing,
    so you must call build_input_with_forcing() before the next model step.

    Example usage:
        # First timestep: initial_state already has forcing
        prediction = model(initial_state)
        state = shift_state_forward(initial_state, prediction)

        # Subsequent timesteps: must add forcing
        model_input = build_input_with_forcing(state, forcing, static)
        prediction = model(model_input)
        state = shift_state_forward(model_input, prediction)

    Channel ordering depends on config['data']['static_first']:

    If static_first == True:
        - Static variables (e.g., Z_GDS4_SFC, LSM) - replicated across time
        - Dynamic forcing (e.g., tsi) - varies per timestep
        - [prognostic + surface + diagnostic variables] - varies per timestep

    If static_first == False (default):
        - Dynamic forcing (e.g., tsi) - varies per timestep
        - Static variables (e.g., Z_GDS4_SFC, LSM) - replicated across time
        - [prognostic + surface + diagnostic variables] - varies per timestep

    Note: Diagnostic variables are OUTPUT ONLY and excluded when shifting state forward.
    """

    def __init__(self, conf: dict) -> None:
        self.conf = conf
        self.history_len = conf["data"]["history_len"]
        self.varnum_diag = len(conf["data"]["diagnostic_variables"])
        self.static_dim = (
            len(conf["data"]["static_variables"])
            if not conf["data"]["static_first"]
            else 0
        )
        self.static_first = conf["data"]["static_first"]

    def shift_state_forward(
        self, state: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Roll the state tensor forward by one timestep.

        Args:
            state: Current state [batch, channels, time, lat, lon]
            prediction: Model prediction for next timestep [batch, channels, 1, lat, lon]

        Returns:
            new_state: State ready for next model call [batch, channels, time, lat, lon]
        """
        if self.history_len == 1:
            # Single timestep history: just return prediction (excluding diagnostics)
            if self.varnum_diag > 0:
                return prediction[:, : -self.varnum_diag, ...].detach()
            else:
                return prediction.detach()
        else:
            # Multi-timestep history: shift time dimension and append new prediction
            if self.static_dim == 0:
                # All variables shift in time
                state_detach = state[:, :, 1:, ...].detach()
            else:
                # Static variables stay fixed, only shift dynamic ones
                state_detach = state[:, : -self.static_dim, 1:, ...].detach()

            # Append new prediction (excluding diagnostic variables)
            if self.varnum_diag > 0:
                new_pred = prediction[:, : -self.varnum_diag, ...].detach()
            else:
                new_pred = prediction.detach()

            return torch.cat([state_detach, new_pred], dim=2)

    def build_input_with_forcing(
        self,
        state: torch.Tensor,
        dynamic_forcing: torch.Tensor,
        static_forcing: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine state with forcing variables to create model input.

        Args:
            state: Current atmospheric state
            dynamic_forcing: Time-varying forcing (e.g., solar radiation)
            static_forcing: Fixed fields (e.g., topography, land-sea mask)

        Returns:
            input_tensor: Ready for model forward pass
        """
        if self.static_first:
            forcing = torch.cat((static_forcing, dynamic_forcing), dim=1)
        else:
            forcing = torch.cat((dynamic_forcing, static_forcing), dim=1)

        return torch.cat((state, forcing), dim=1)


# ============================================================================
# CAMULATOR STEPPER - THE CORE INTERFACE FOR COUPLING
# ============================================================================


class CAMulatorStepper:
    """
    Core CAMulator time-stepping interface suitable for coupling to other models.

    This class isolates the physics integration step from I/O, initialization,
    and post-processing setup. Suitable for coupling to ocean models or other
    Earth system components.

    Usage:
        stepper = CAMulatorStepper(model, conf, device)

        for timestep in range(num_steps):
            # Get forcing for this timestep from your coupler
            dynamic_forcing = get_dynamic_forcing(timestep)

            # Step the atmosphere forward
            prediction = stepper.step(state, dynamic_forcing, static_forcing)

            # Update state for next step
            state = stepper.state_manager.shift_state_forward(state, prediction)

    Attributes:
        model: PyTorch model for forward integration
        state_manager: StateManager instance for state transformations
        device: torch.device for computation
    """

    def __init__(self, model: torch.nn.Module, conf: dict, device: torch.device):
        """
        Initialize the CAMulator stepper with full post-processing.

        Args:
            model: Trained CAMulator model (already on device, in eval mode)
            conf: Full configuration dictionary
            device: Device for computation (cuda/cpu)
        """
        self.model = model
        self.conf = conf
        self.device = device
        self.state_manager = StateManager(conf)

        # Create variable accessors for convenience
        self.state_accessor = StateVariableAccessor(conf, tensor_type="state")
        self.input_accessor = StateVariableAccessor(conf, tensor_type="input")
        self.output_accessor = StateVariableAccessor(conf, tensor_type="output")

        # Setup post-processing components (conservation fixers, wind filtering)
        self._setup_postprocessing()

    def _setup_postprocessing(self) -> None:
        """
        Initialize post-processing components (conservation fixers and wind filtering).

        Sets up:
        - Global mass fixer (dry air mass conservation)
        - Global water fixer (water mass conservation)
        - Global energy fixer (total energy conservation)
        - Wind artifact filtering flags
        """
        post_conf = self.conf["model"]["post_conf"]
        postblock_available = _load_postblock_modules()
        windpp_available = _load_windpp_module()

        # Check which conservation fixers are enabled
        self.flag_mass = (
            postblock_available
            and post_conf["activate"]
            and post_conf["global_mass_fixer"]["activate"]
        )
        self.flag_water = (
            postblock_available
            and post_conf["activate"]
            and post_conf["global_water_fixer"]["activate"]
        )
        self.flag_energy = (
            postblock_available
            and post_conf["activate"]
            and post_conf["global_energy_fixer"]["activate"]
        )

        # Initialize conservation fixers
        if self.flag_mass:
            self.opt_mass = GlobalMassFixer(post_conf)
            logger.info("Global mass fixer initialized")
        if self.flag_water:
            self.opt_water = GlobalWaterFixer(post_conf)
            logger.info("Global water fixer initialized")
        if self.flag_energy:
            self.opt_energy = GlobalEnergyFixer(post_conf)
            logger.info("Global energy fixer initialized")

        # Wind filtering flag
        self.enable_wind_filtering = windpp_available

    def step(
        self,
        state: torch.Tensor,
        dynamic_forcing: torch.Tensor,
        static_forcing: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance the atmospheric state by one model timestep.

        This is the core coupling interface - a pure function that takes
        atmospheric state and returns the next state with full physics
        post-processing applied.

        Args:
            state: Current atmospheric state [batch, state_channels, time, lat, lon]
            dynamic_forcing: Time-varying forcing [batch, dyn_channels, time, lat, lon]
            static_forcing: Static forcing [batch, static_channels, time, lat, lon]

        Returns:
            prediction: Next atmospheric state [batch, output_channels, 1, lat, lon]
                       Includes prognostic, surface, and diagnostic variables with
                       conservation fixers and wind filtering applied
        """
        # Build model input by combining state with forcing
        model_input = self.state_manager.build_input_with_forcing(
            state, dynamic_forcing, static_forcing
        )

        # Run model inference
        with torch.no_grad():
            prediction = self.model(model_input.float())

        # Apply post-processing (conservation fixers and wind filtering)
        prediction = self._apply_postprocessing(prediction, model_input)

        return prediction

    def _apply_postprocessing(
        self, prediction: torch.Tensor, model_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply wind artifact filtering and conservation fixers.

        Post-processing order:
        1. Wind artifact filtering (removes spurious wind patterns near surface)
        2. Global mass fixer (conserves dry air mass)
        3. Global water fixer (conserves total water mass)
        4. Global energy fixer (conserves total energy)

        Args:
            prediction: Raw model prediction [batch, output_channels, 1, lat, lon]
            model_input: Model input tensor (needed for some fixers)

        Returns:
            prediction: Post-processed prediction with conservation applied
        """
        # Wind artifact filtering
        if self.enable_wind_filtering:
            post_process_wind_artifacts(prediction, self.conf, enable_filtering=True)

        # Apply conservation fixers in sequence
        if self.flag_mass:
            prediction = self.opt_mass({"y_pred": prediction, "x": model_input})[
                "y_pred"
            ]

        if self.flag_water:
            prediction = self.opt_water({"y_pred": prediction, "x": model_input})[
                "y_pred"
            ]

        if self.flag_energy:
            prediction = self.opt_energy({"y_pred": prediction, "x": model_input})[
                "y_pred"
            ]

        return prediction

    def get_state_var(
        self,
        tensor: torch.Tensor,
        var_name: str,
        tensor_type: Literal["state", "input", "output"] = "state",
        time_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Convenience method to get a variable from any tensor type.

        Args:
            tensor: Tensor to extract from
            var_name: Variable name
            tensor_type: Type of tensor ('state', 'input', or 'output')
            time_idx: Optional time index

        Returns:
            Variable tensor
        """
        accessor = {
            "state": self.state_accessor,
            "input": self.input_accessor,
            "output": self.output_accessor,
        }[tensor_type]

        return accessor.get_state_var(tensor, var_name, time_idx)

    def set_state_var(
        self,
        tensor: torch.Tensor,
        var_name: str,
        var_data: torch.Tensor,
        tensor_type: Literal["state", "input", "output"] = "state",
        time_idx: Optional[int] = None,
    ) -> None:
        """
        Convenience method to set a variable in any tensor type.

        Args:
            tensor: Tensor to modify
            var_name: Variable name
            var_data: New data
            tensor_type: Type of tensor ('state', 'input', or 'output')
            time_idx: Optional time index
        """
        accessor = {
            "state": self.state_accessor,
            "input": self.input_accessor,
            "output": self.output_accessor,
        }[tensor_type]

        accessor.set_state_var(tensor, var_name, var_data, time_idx)


# ============================================================================
# INITIALIZATION - ONE-TIME SETUP FOR CAMULATOR
# ============================================================================


def initialize_camulator(
    config_path: str,
    model_name: Optional[str] = None,
    device: str = "cuda",
    logger: LoggerLike | None = None,
) -> dict:
    """
    One-time initialization of CAMulator model and all supporting components.

    This function loads the model, transforms, forcing data, and sets up everything
    needed for CAMulator integration. Separate from the time-stepping loop for
    cleaner coupling interfaces.

    Args:
        config_path: Path to YAML configuration file
        model_name: Optional specific checkpoint name (e.g., 'checkpoint.pt00091.pt')
                   If None, uses default 'checkpoint.pt'
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        context: Dictionary containing all initialized components:
            - 'model': Loaded CAMulator model in eval mode
            - 'stepper': CAMulatorStepper instance ready for time-stepping
            - 'conf': Parsed configuration dictionary
            - 'state_transformer': Normalization/denormalization transforms
            - 'forcing_dataset': xarray dataset with normalized forcing data
            - 'static_forcing': Static forcing tensor (topography, LSM, etc.)
            - 'initial_state': Initial condition tensor loaded from file
            - 'latlons': Latitude/longitude coordinates
            - 'metadata': Variable metadata for NetCDF output
            - 'device': torch device object

    Example:
        >>> ctx = initialize_camulator('camulator_config.yml',
        ...                           model_name='checkpoint.pt00091.pt')
        >>> stepper = ctx['stepper']
        >>> state = ctx['initial_state']
        >>>
        >>> # Run simulation
        >>> for timestep in range(num_steps):
        ...     dynamic_forcing = get_forcing(timestep, ctx['forcing_dataset'])
        ...     prediction = stepper.step(state, dynamic_forcing, ctx['static_forcing'])
        ...     state = stepper.state_manager.shift_state_forward(state, prediction)

    Raises:
        FileNotFoundError: If config file or checkpoint not found
        ImportError: If required CREDIT modules not available
    """
    _load_credit_modules()

    log = logger if logger is not None else get_default_logger()
    log.info(f"Initializing CAMulator from config: {config_path}")

    # Load and parse configuration
    with open(config_path) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=False, parse_predict=True, print_summary=False
    )
    conf["predict"]["mode"] = None  # Override to None for single-GPU inference

    current_device = torch.device(device)
    log.info(f"Using device: {current_device}")

    # Load transforms and normalization
    log.info("Loading transforms...")
    transform = load_transforms(conf)  # noqa: F841

    if conf["data"]["scaler_type"] == "std_new":
        state_transformer = Normalize_ERA5_and_Forcing(conf)
    else:
        raise ValueError(f"Unsupported scaler_type: {conf['data']['scaler_type']}")

    # Load model
    log.info(
        f"Loading model: {model_name if model_name else 'checkpoint.pt (default)'}"
    )
    if model_name:
        model = load_model_name(conf, model_name, load_weights=True).to(current_device)
    else:
        model = load_model(conf, load_weights=True).to(current_device)

    # Handle distributed mode if specified (typically not used for climate runs)
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:
        log.info(f"Setting up distributed mode: {conf['predict']['mode']}")
        model = distributed_model_wrapper(conf, model, current_device)
        if conf["predict"]["mode"] == "fsdp":
            model = load_model_state(conf, model, current_device)

    model.eval()
    log.info("Model loaded and set to eval mode")

    # Load initial conditions
    log.info("Loading initial conditions...")
    ic_path = conf["predict"]["init_cond_fast_climate"]
    if not os.path.exists(ic_path):
        raise FileNotFoundError(
            f"Initial condition file not found: {ic_path}\nPlease run Make_Climate_Initial_Conditions.py first."
        )

    initial_state = torch.load(ic_path, map_location=current_device).to(current_device)
    log.info(f"Initial state shape: {initial_state.shape}")

    # Load forcing data
    log.info("Loading forcing data...")
    forcing_file = conf["predict"]["forcing_file"]
    if not os.path.exists(forcing_file):
        raise FileNotFoundError(f"Forcing file not found: {forcing_file}")

    chunk_size = conf["data"].get("forcing_chunk_size", 32)
    forcing_ds = xr.open_dataset(forcing_file, chunks={"time": chunk_size})

    # Normalize forcing data
    log.info("Normalizing forcing data...")
    forcing_ds_norm = state_transformer.transform_dataset(forcing_ds)
    forcing_ds_norm = forcing_ds_norm.chunk({"time": chunk_size})

    # Load static forcing (topography, land-sea mask, etc.)
    log.info("Loading static forcing...")
    sf_vars = conf["data"]["static_variables"]
    static_forcing = _prepare_static_forcing_tensor(forcing_ds, sf_vars, current_device)
    log.info(f"Static forcing shape: {static_forcing.shape}")

    # Load metadata and coordinates
    log.info("Loading metadata and coordinates...")
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])
    metadata = load_metadata(conf)

    # Create CAMulatorStepper with full post-processing
    log.info("Creating CAMulatorStepper with conservation fixers...")
    stepper = CAMulatorStepper(model, conf, current_device)

    log.info("=" * 70)
    log.info("Initialization complete!")
    log.info(f"Model device: {current_device}")
    log.info(f"State shape: {initial_state.shape}")
    log.info(f"Static forcing: {len(sf_vars)} variables")
    log.info(
        f"Conservation fixers: Mass={stepper.flag_mass}, Water={stepper.flag_water}, Energy={stepper.flag_energy}"
    )
    log.info(f"Wind filtering: {stepper.enable_wind_filtering}")
    log.info("=" * 70)

    return {
        "model": model,
        "stepper": stepper,
        "conf": conf,
        "state_transformer": state_transformer,
        "forcing_dataset": forcing_ds_norm,
        "forcing_dataset_raw": forcing_ds.chunk({"time": chunk_size}),
        "static_forcing": static_forcing,
        "initial_state": initial_state,
        "latlons": latlons,
        "metadata": metadata,
        "device": current_device,
    }
