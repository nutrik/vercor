from __future__ import annotations

from pathlib import Path

import vercor.jax_logging as jax_logging_module
from tests._architecture_support import package_import_cycles, source_for

EXPECTED_PUBLIC_LOGGING_NAMES = [
    "CANONICAL_LOG_DATE_FORMAT",
    "CANONICAL_LOG_FORMAT",
    "DEFAULT_LOGGER_NAME",
    "JaxCallbackLogger",
    "LoggerLike",
    "configure_python_logger",
    "effective_log_level",
    "emit_host_log",
    "get_default_logger",
    "logger_enabled_for",
    "normalize_log_level",
    "setup_logger",
]


def test_jax_logging_module_is_thin_public_facade() -> None:
    source = source_for("vercor/jax_logging.py")

    assert "class JaxCallbackLogger" not in source
    assert "jax.debug.callback" not in source
    assert "from vercor._logging." in source


def test_private_logging_package_owns_focused_implementation_modules() -> None:
    expected_paths = [
        Path("vercor/_logging/__init__.py"),
        Path("vercor/_logging/protocols.py"),
        Path("vercor/_logging/config.py"),
        Path("vercor/_logging/host.py"),
        Path("vercor/_logging/callback.py"),
    ]

    for path in expected_paths:
        assert path.exists(), path

    assert package_import_cycles("vercor/_logging", "vercor._logging") == []


def test_production_code_uses_jax_logging_public_facade_only() -> None:
    for path in Path("vercor").rglob("*.py"):
        if path == Path("vercor/jax_logging.py") or path.parts[:2] == (
            "vercor",
            "_logging",
        ):
            continue

        source = path.read_text(encoding="utf-8")
        assert "vercor._logging" not in source, path


def test_jax_logging_facade_preserves_public_api() -> None:
    assert jax_logging_module.__all__ == EXPECTED_PUBLIC_LOGGING_NAMES
    for name in EXPECTED_PUBLIC_LOGGING_NAMES:
        assert hasattr(jax_logging_module, name), name
