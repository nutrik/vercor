"""Veros ocean component factory."""

from __future__ import annotations

from functools import partial
from typing import Any

from vercor.components import (
    CallableComponent,
    Component,
    LifecycleHooks,
    ComponentSpec,
)
from vercor.output import OutputSpec
from vercor.setups.config import VerosConfig


def _load_veros_implementation() -> tuple[Any, Any, Any, type[Any]]:
    """Load Veros implementation owners after runtime configuration."""

    import vercor.setups._external.veros_gcm_state as veros_gcm_state
    import vercor.setups._external.veros_output as veros_output
    import vercor.setups._external.veros_runtime as veros_runtime
    from vercor.setups._external.veros_gcm_state import VerosGCMSetupState

    return veros_gcm_state, veros_output, veros_runtime, VerosGCMSetupState


def make_veros_gcm(
    *,
    config: VerosConfig | None = None,
) -> Component:
    """Return a host-backed or differentiable JAX Veros GCM component."""

    try:
        import veros  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "The VerosGCM component requires the Veros package. Please install it "
            "with `pip install veros`."
        ) from error

    from vercor.setups._external import veros_runtime_settings

    config = VerosConfig() if config is None else config
    veros_runtime_settings.configure_veros_runtime(config.execution)
    if config.execution == "jax":
        veros_runtime_settings.require_differentiable_veros_capabilities()

    (
        _veros_gcm_state,
        _veros_output,
        _veros_runtime,
        VerosGCMSetupState,
    ) = _load_veros_implementation()

    state = VerosGCMSetupState(
        name=config.name,
        setup=config.setup,
        execution=config.execution,
        spinup_time=config.spinup.duration,
        custom_parameters=config.custom_parameters,
        restore_to_climatology=config.restore_to_climatology,
        do_spinup=config.spinup.enabled,
    )
    component = CallableComponent(
        config.name,
        state.grid,
        partial(_veros_runtime.step_veros_runtime, state),
        spec=ComponentSpec(
            inputs=(
                _veros_gcm_state.VEROS_INPUT_FIELD_NAMES
                if state.uses_atmosphere_forcing
                else ()
            ),
            outputs=("sea_surface_temperature",),
            initial_fields=_veros_gcm_state.veros_default_fields(),
            execution=config.execution,
            lifecycle=LifecycleHooks(setup=state.setup),
            output=OutputSpec(
                provider=(
                    _veros_output.veros_output_provider()
                    if config.output.provider is None
                    else config.output.provider
                ),
                snapshot_writer=config.output.snapshot_writer
                or _veros_output.write_veros_snapshot_output,
                period=config.output.period,
            ),
        ),
    )
    return component


__all__ = [
    "make_veros_gcm",
]
