"""Lazy optional-dependency loading for CAMulator setup adapters."""

from __future__ import annotations

from typing import Any

CREDIT_AVAILABLE = False
POSTBLOCK_AVAILABLE = False

load_model: Any = None
load_model_name: Any = None
load_transforms: Any = None
Normalize_ERA5_and_Forcing: Any = None
distributed_model_wrapper: Any = None
load_model_state: Any = None
credit_main_parser: Any = None

GlobalMassFixer: Any = None
GlobalWaterFixer: Any = None
GlobalEnergyFixer: Any = None


def load_credit_modules() -> None:
    """Load CREDIT core modules at the CAMulator execution boundary."""

    global CREDIT_AVAILABLE
    global load_model, load_model_name, load_transforms, Normalize_ERA5_and_Forcing
    global distributed_model_wrapper, load_model_state, credit_main_parser

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
    CREDIT_AVAILABLE = True


def load_postblock_modules() -> bool:
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


__all__ = [
    "CREDIT_AVAILABLE",
    "POSTBLOCK_AVAILABLE",
    "GlobalEnergyFixer",
    "GlobalMassFixer",
    "GlobalWaterFixer",
    "Normalize_ERA5_and_Forcing",
    "load_credit_modules",
    "load_postblock_modules",
    "credit_main_parser",
    "distributed_model_wrapper",
    "load_model",
    "load_model_name",
    "load_model_state",
    "load_transforms",
]
