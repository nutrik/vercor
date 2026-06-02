from __future__ import annotations

import logging

DEFAULT_LOGGER_NAME = "VerCOR"
CANONICAL_LOG_FORMAT = (
    f"{DEFAULT_LOGGER_NAME}: %(asctime)s [%(levelname)s]: %(message)s"
)
CANONICAL_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CANONICAL_HANDLER_MARKER = "_vercor_canonical_handler"


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
