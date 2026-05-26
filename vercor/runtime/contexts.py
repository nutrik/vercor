from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vercor.calendar import ModelDateTime
from vercor.jax_logging import LoggerLike
from vercor.run_sequence import RunSequence
from vercor.settings import VercorSettings


@dataclass(frozen=True)
class ComponentInitContext:
    """Minimal component initialization context owned by the coupler."""

    start: datetime | ModelDateTime
    dt_seconds: float
    run_sequence: RunSequence
    settings: VercorSettings
    logger: LoggerLike


@dataclass(frozen=True)
class RuntimeStepContext:
    """Minimal runtime step context passed to component step boundaries."""

    dt_seconds: float
    settings: VercorSettings
    time: datetime | ModelDateTime | None = None
    logger: LoggerLike | None = None
