from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from vercor.calendar import ModelDateTime
from vercor.jax_logging import LoggerLike
from vercor.settings import VercorSettings


@dataclass(frozen=True)
class ComponentSetupContext:
    """Minimal setup context passed to component initialization hooks."""

    start: datetime | ModelDateTime
    dt_seconds: float
    run_sequence: Sequence[str]
    settings: VercorSettings
    logger: LoggerLike


@dataclass(frozen=True)
class ComponentStepContext:
    """Minimal runtime step context passed to component step boundaries."""

    dt_seconds: float
    settings: VercorSettings
    time: datetime | ModelDateTime | None = None
    logger: LoggerLike | None = None


__all__ = ["ComponentSetupContext", "ComponentStepContext"]
