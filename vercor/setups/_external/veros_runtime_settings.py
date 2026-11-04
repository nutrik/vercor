"""Explicit host and JAX runtime configuration for the Veros adapter."""

from __future__ import annotations

from typing import Any, Literal


def _set_runtime_setting(runtime_settings: Any, name: str, value: Any) -> None:
    """Set one Veros runtime setting, tolerating already-applied locked values."""

    try:
        setattr(runtime_settings, name, value)
    except RuntimeError:
        if getattr(runtime_settings, name, None) == value:
            return
        current = getattr(runtime_settings, name, None)
        raise RuntimeError(
            "Veros runtime settings are already locked with "
            f"{name}={current!r}; requested {value!r}. Construct host and JAX "
            "Veros components in separate Python processes."
        ) from None


def configure_veros_runtime(
    execution: Literal["host", "jax"] = "host",
    *,
    backend: Literal["numpy", "jax"] | None = None,
) -> None:
    """Select native operators independently of host or scanned execution.

    Stock Veros supports JAX kernels under host orchestration. Scanned execution
    additionally requires the differentiable fork and cannot use NumPy kernels.
    None preserves the execution-based native choice.
    """
    if execution not in ("host", "jax"):
        raise ValueError("execution must be 'host' or 'jax'")
    if backend not in (None, "numpy", "jax"):
        raise ValueError("backend must be 'numpy', 'jax', or None")
    if execution == "jax" and backend == "numpy":
        raise ValueError("JAX execution requires the native JAX backend")
    from veros import runtime_settings  # type: ignore[import]

    backend = backend or ("numpy" if execution == "host" else "jax")
    linear_solver = "best" if execution == "host" else "scipy_jax"
    _set_runtime_setting(runtime_settings, "backend", backend)
    _set_runtime_setting(runtime_settings, "linear_solver", linear_solver)
    if execution == "host":
        _set_runtime_setting(runtime_settings, "force_overwrite", True)
        _set_runtime_setting(runtime_settings, "diskless_mode", True)


def require_differentiable_veros_capabilities() -> None:
    """Require the local differentiable Veros fork's traced-state features."""

    from veros import variables as veros_variables  # type: ignore[import]
    from veros.core import operators  # type: ignore[import]
    from veros.state import VerosState  # type: ignore[import]

    missing: list[str] = []
    if not callable(getattr(VerosState, "copy", None)):
        missing.append("VerosState.copy")
    if not callable(getattr(operators, "safe_sqrt", None)):
        missing.append("veros.core.operators.safe_sqrt")
    variable_registry = getattr(veros_variables, "VARIABLES", {})
    for variable_name in ("c_k", "c_eps"):
        if variable_name not in variable_registry:
            missing.append(f"variables.{variable_name}")
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "JAX execution requires the differentiable Veros fork at commit "
            "7a8c964; missing capabilities: "
            f"{missing_text}. Install the approved local vercor_etienne/veros "
            "checkout in this Python environment."
        )


__all__ = [
    "configure_veros_runtime",
    "require_differentiable_veros_capabilities",
]
