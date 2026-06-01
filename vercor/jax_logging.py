from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax

DEFAULT_LOGGER_NAME = "VerCOR"
CANONICAL_LOG_FORMAT = (
    f"{DEFAULT_LOGGER_NAME} ─ %(asctime)s [%(levelname)s]: %(message)s"
)
CANONICAL_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CANONICAL_HANDLER_MARKER = "_vercor_canonical_handler"


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
    if isinstance(logger, JaxCallbackLogger):
        logger.logger.log(level, formatted)
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


def normalize_log_level(level: int | str) -> int:
    """Return a standard ``logging`` integer level from a string or integer."""

    if isinstance(level, str):
        normalized = logging.getLevelName(level.upper())
        if not isinstance(normalized, int):
            raise ValueError(f"Unknown logging level: {level}")
        return normalized
    return int(level)


def get_default_logger() -> logging.Logger:
    """Return the default Python logger used for VerCOR host-side messages."""

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    level = logger.level if logger.level != logging.NOTSET else logging.INFO
    return configure_python_logger(logger, level)


def configure_python_logger(
    logger: logging.Logger,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Configure ``logger`` to emit VerCOR records with the canonical format."""

    normalized_level = normalize_log_level(level)
    default_logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    _install_canonical_handler(default_logger)
    default_logger.propagate = False

    if logger.name == DEFAULT_LOGGER_NAME:
        logger.setLevel(normalized_level)
        return logger

    logger.setLevel(normalized_level)
    if logger.name.startswith(f"{DEFAULT_LOGGER_NAME}."):
        _remove_noncanonical_handlers(logger)
        logger.propagate = True
        return logger

    _install_canonical_handler(logger)
    logger.propagate = False
    return logger


def effective_log_level(logger: LoggerLike, default: int | str = logging.INFO) -> int:
    """Return the effective level for logger-like objects."""

    get_effective_level = getattr(logger, "getEffectiveLevel", None)
    if callable(get_effective_level):
        return int(get_effective_level())

    level = getattr(logger, "level", None)
    if isinstance(level, (int, str)):
        return normalize_log_level(level)

    return normalize_log_level(default)


@dataclass
class JaxCallbackLogger:
    """Small logger wrapper that emits messages through ``jax.debug.callback``."""

    logger: logging.Logger

    @property
    def name(self) -> str:
        """Return the wrapped Python logger name."""

        return self.logger.name

    @property
    def level(self) -> int:
        """Return the wrapped Python logger level."""

        return self.logger.level

    def getEffectiveLevel(self) -> int:
        """Return the effective logging threshold."""

        return self.logger.getEffectiveLevel()

    def setLevel(self, level: int | str) -> None:
        """Set the wrapped Python logger threshold."""

        self.logger.setLevel(normalize_log_level(level))

    def isEnabledFor(self, level: int) -> bool:
        """Return whether ``level`` is enabled on the wrapped logger."""

        return self.logger.isEnabledFor(level)

    def debug(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Emit a debug message through a JAX callback."""

        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Emit an informational message through a JAX callback."""

        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Emit a warning message through a JAX callback."""

        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Emit an error message through a JAX callback."""

        self._log(logging.ERROR, message, *args, **kwargs)

    def _log(self, level: int, message: object, *args: Any, **kwargs: Any) -> None:
        if not logger_enabled_for(self.logger, level):
            return

        static_args, dynamic_args, dynamic_arg_indices = _partition_dynamic(args)
        static_kwargs, dynamic_kwargs, dynamic_kwarg_names = _partition_dynamic_kwargs(
            kwargs
        )

        def emit(*callback_args: Any, **callback_kwargs: Any) -> None:
            formatted_args = list(static_args)
            for index, value in zip(dynamic_arg_indices, callback_args):
                formatted_args[index] = _host_value(value)
            formatted_kwargs = dict(static_kwargs)
            for name in dynamic_kwarg_names:
                formatted_kwargs[name] = _host_value(callback_kwargs[name])
            emit_host_log(
                self,
                level,
                message,
                *tuple(formatted_args),
                **formatted_kwargs,
            )

        jax.debug.callback(
            emit,
            *dynamic_args,
            ordered=True,
            **dynamic_kwargs,
        )


def setup_logger(
    level: int | str = logging.INFO,
    name: str = DEFAULT_LOGGER_NAME,
) -> JaxCallbackLogger:
    """Set up and return the callback-backed VerCOR logger."""

    logger = (
        get_default_logger() if name == DEFAULT_LOGGER_NAME else logging.getLogger(name)
    )
    logger = configure_python_logger(logger, level)
    return JaxCallbackLogger(logger)


def _install_canonical_handler(logger: logging.Logger) -> None:
    _remove_noncanonical_handlers(logger)
    for handler in logger.handlers:
        if getattr(handler, _CANONICAL_HANDLER_MARKER, False):
            handler.setFormatter(
                logging.Formatter(CANONICAL_LOG_FORMAT, CANONICAL_LOG_DATE_FORMAT)
            )
            return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(CANONICAL_LOG_FORMAT, CANONICAL_LOG_DATE_FORMAT)
    )
    setattr(handler, _CANONICAL_HANDLER_MARKER, True)
    logger.addHandler(handler)


def _remove_noncanonical_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if not getattr(handler, _CANONICAL_HANDLER_MARKER, False):
            logger.removeHandler(handler)


def _partition_dynamic(
    values: tuple[Any, ...],
) -> tuple[list[Any], list[Any], list[int]]:
    static_values: list[Any] = []
    dynamic_values: list[Any] = []
    dynamic_indices: list[int] = []
    for index, value in enumerate(values):
        if _is_dynamic_callback_value(value):
            static_values.append(None)
            dynamic_values.append(value)
            dynamic_indices.append(index)
        else:
            static_values.append(value)
    return static_values, dynamic_values, dynamic_indices


def _partition_dynamic_kwargs(
    values: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    static_values: dict[str, Any] = {}
    dynamic_values: dict[str, Any] = {}
    dynamic_names: list[str] = []
    for name, value in values.items():
        if _is_dynamic_callback_value(value):
            dynamic_values[name] = value
            dynamic_names.append(name)
        else:
            static_values[name] = value
    return static_values, dynamic_values, dynamic_names


def _is_dynamic_callback_value(value: Any) -> bool:
    return isinstance(
        value,
        (
            jax.Array,
            jax.core.Tracer,
        ),
    )


def _host_value(value: Any) -> Any:
    host_value = jax.device_get(value)
    if getattr(host_value, "shape", None) == ():
        return host_value.item()
    return host_value


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
