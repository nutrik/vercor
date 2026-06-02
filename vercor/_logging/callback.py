from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax

from vercor._logging.config import (
    DEFAULT_LOGGER_NAME,
    configure_python_logger,
    get_default_logger,
    normalize_log_level,
)
from vercor._logging.host import emit_host_log
from vercor._logging.protocols import logger_enabled_for


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
