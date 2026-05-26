"""Compatibility facade for focused CAMulator setup modules.

New code should import optional dependency loaders, forcing cursors, tensor
helpers, steppers, and initialization from their focused modules.
"""

from __future__ import annotations

from vercor.setups.external.camulator_forcing import (
    CAMulatorForcingCursor,
    CamulatorRuntimeCursor,
    initialize_camulator_forcing_cursor,
    load_camulator_forcing_context,
    parse_datetime_from_config,
)
from vercor.setups.external.camulator_imports import (
    CREDIT_AVAILABLE,
    POSTBLOCK_AVAILABLE,
    WINDPP_AVAILABLE,
    GlobalEnergyFixer,
    GlobalMassFixer,
    GlobalWaterFixer,
    Normalize_ERA5_and_Forcing,
    _load_credit_modules,
    _load_postblock_modules,
    _load_windpp_module,
    credit_main_parser,
    distributed_model_wrapper,
    load_metadata,
    load_model,
    load_model_name,
    load_model_state,
    load_transforms,
    post_process_wind_artifacts,
)
from vercor.setups.external.camulator_init import initialize_camulator
from vercor.setups.external.camulator_stepper import CAMulatorStepper, StateManager
from vercor.setups.external.camulator_tensors import (
    StateVariableAccessor,
    _append_indexed_variables,
    _mark_unavailable_variables,
    _prepare_static_forcing_tensor,
)

__all__ = [
    "CAMulatorForcingCursor",
    "CAMulatorStepper",
    "CREDIT_AVAILABLE",
    "CamulatorRuntimeCursor",
    "GlobalEnergyFixer",
    "GlobalMassFixer",
    "GlobalWaterFixer",
    "Normalize_ERA5_and_Forcing",
    "POSTBLOCK_AVAILABLE",
    "StateManager",
    "StateVariableAccessor",
    "WINDPP_AVAILABLE",
    "_append_indexed_variables",
    "_load_credit_modules",
    "_load_postblock_modules",
    "_load_windpp_module",
    "_mark_unavailable_variables",
    "_prepare_static_forcing_tensor",
    "credit_main_parser",
    "distributed_model_wrapper",
    "initialize_camulator",
    "initialize_camulator_forcing_cursor",
    "load_camulator_forcing_context",
    "load_metadata",
    "load_model",
    "load_model_name",
    "load_model_state",
    "load_transforms",
    "parse_datetime_from_config",
    "post_process_wind_artifacts",
]
