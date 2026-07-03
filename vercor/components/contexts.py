from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from vercor._deprecation import deprecated_getattr
from vercor.calendar import ModelDateTime
from vercor.jax_logging import LoggerLike
from vercor.settings import VercorSettings


@dataclass(frozen=True)
class SetupContext:
    """Minimal setup context passed to component initialization hooks."""

    start: datetime | ModelDateTime
    dt_seconds: float
    run_sequence: Sequence[str]
    settings: VercorSettings
    logger: LoggerLike


@dataclass(frozen=True)
class StepContext:
    """Minimal runtime step context passed to component step boundaries."""

    dt_seconds: float
    settings: VercorSettings
    time: datetime | ModelDateTime | None = None
    logger: LoggerLike | None = None
    step: int = 0


__all__ = [
    "SetupContext",
    "StepContext",
]


__getattr__ = deprecated_getattr(
    __name__,
    {
        "ComponentSetupContext": (
            "vercor.components.contexts.SetupContext",
            SetupContext,
        ),
        "ComponentStepContext": ("vercor.components.contexts.StepContext", StepContext),
    },
    remove_in="0.2.0",
)
