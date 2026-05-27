"""Explicit Veros host-runtime configuration used by the Veros adapter."""


def configure_veros_runtime() -> None:
    """Configure Veros for the NumPy host runtime used by ``VerosGCM``."""

    from veros import runtime_settings  # type: ignore[import]

    try:
        setattr(runtime_settings, "backend", "numpy")
        setattr(runtime_settings, "force_overwrite", True)
    except RuntimeError:
        if getattr(runtime_settings, "backend", None) == "numpy":
            return
        raise
