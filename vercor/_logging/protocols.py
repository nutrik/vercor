from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from vercor._logging.config import normalize_log_level


@runtime_checkable
class LoggerLike(Protocol):
    """Logger interface used across Python and JAX callback runtimes."""

    def debug(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Emit a debug message."""

    def info(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Emit an informational message."""

    def warning(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Emit a warning message."""

    def error(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Emit an error message."""

    def setLevel(self, level: int | str) -> None:
        """Set the logger threshold."""

    def isEnabledFor(self, level: int) -> bool:
        """Return whether a level is enabled."""


def logger_enabled_for(logger: Any, level: int) -> bool:
    """Return whether ``logger`` should emit ``level`` host-side messages."""

    is_enabled_for = getattr(logger, "isEnabledFor", None)
    if callable(is_enabled_for):
        return bool(is_enabled_for(level))
    return True


def effective_log_level(logger: LoggerLike, default: int | str = logging.INFO) -> int:
    """Return the effective level for logger-like objects."""

    get_effective_level = getattr(logger, "getEffectiveLevel", None)
    if callable(get_effective_level):
        return int(get_effective_level())

    level = getattr(logger, "level", None)
    if isinstance(level, (int, str)):
        return normalize_log_level(level)

    return normalize_log_level(default)
