from __future__ import annotations

from collections.abc import Sequence
import logging

import jax

from vercor.clock import Clock
from vercor.jax_logging import (
    LoggerLike,
    emit_host_log,
    logger_enabled_for,
)
from vercor.types import RuntimeArray


def runtime_step_progress_message(n: int, time: object, dt: object) -> str:
    """Return the shared host/scanned runtime step progress message."""

    return f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} "


def runtime_component_progress_message(component_name: str) -> str:
    """Return the shared host/scanned runtime component progress message."""

    return f" Run component: {component_name}"


def runtime_step_progress_messages(clock: Clock) -> tuple[str, ...]:
    """Return host-rendered progress messages for all scanned runtime steps."""

    return tuple(
        runtime_step_progress_message(n, time, dt) for n, time, dt in clock.iter()
    )


def log_scanned_step_progress(
    logger: LoggerLike,
    step_index: RuntimeArray,
    step_progress_messages: Sequence[str],
) -> None:
    """Emit one scanned-runtime step progress message through a host callback."""

    if not logger_enabled_for(logger, logging.INFO):
        return

    def emit(index: RuntimeArray) -> None:
        host_index = int(jax.device_get(index).item())
        emit_host_log(
            logger,
            logging.INFO,
            step_progress_messages[host_index],
        )

    jax.debug.callback(emit, step_index, ordered=True)


def log_scanned_component_progress(
    logger: LoggerLike,
    component_name: str,
) -> None:
    """Emit one scanned-runtime component progress message through a callback."""

    if not logger_enabled_for(logger, logging.INFO):
        return

    jax.debug.callback(
        lambda: emit_host_log(
            logger,
            logging.INFO,
            runtime_component_progress_message(component_name),
        ),
        ordered=True,
    )
