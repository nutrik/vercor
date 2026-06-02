from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from vercor._logging.protocols import logger_enabled_for


def emit_host_log(
    logger: Any,
    level: int,
    message: object,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Emit a formatted log record on the host without adding a JAX callback."""

    if not logger_enabled_for(logger, level):
        return

    formatted = _format_message(message, args, kwargs)
    wrapped_logger = getattr(logger, "logger", None)
    if isinstance(wrapped_logger, logging.Logger):
        wrapped_logger.log(level, formatted)
        return

    log = getattr(logger, "log", None)
    if callable(log):
        log(level, formatted)
        return

    level_name = logging.getLevelName(level)
    if isinstance(level_name, str):
        method = getattr(logger, level_name.lower(), None)
        if callable(method):
            method(formatted)


def _format_message(
    message: object,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> str:
    template = str(message)
    if not args and not kwargs:
        return template
    try:
        return template.format(*args, **kwargs)
    except (IndexError, KeyError, ValueError):
        if args and not kwargs:
            try:
                return template % args
            except (TypeError, ValueError):
                pass
        return " ".join([template, *(str(arg) for arg in args)])
