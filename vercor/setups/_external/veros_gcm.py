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
from vercor.fields import _flatten_field_items
from vercor.recipes import VERIS_TO_OCEAN_FLUX_FIELDS, OCEAN_TO_VERIS_FORCING_FIELDS
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

    config = VerosConfig() if config is None else config
    return _make_veros_component(_initialize_veros_owner(config), config)


def _initialize_veros_owner(config: VerosConfig) -> Any:
    """Initialize the sole native ocean owner after configuring its runtime."""
    try:
        import veros  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "The VerosGCM component requires the Veros package. Please install it "
            "with `pip install veros`."
        ) from error

    from vercor.setups._external import veros_runtime_settings

    veros_runtime_settings.configure_veros_runtime(config.execution)
    if config.execution == "jax":
        veros_runtime_settings.require_differentiable_veros_capabilities()

    (
        _veros_gcm_state,
        _veros_output,
        _veros_runtime,
        VerosGCMSetupState,
    ) = _load_veros_implementation()

    return VerosGCMSetupState(
        name=config.name,
        setup=config.setup,
        execution=config.execution,
        spinup_time=config.spinup.duration,
        custom_parameters=config.custom_parameters,
        restore_to_climatology_temp=config.restore_to_climatology_temp,
        restore_to_climatology_salt=config.restore_to_climatology_salt,
        do_spinup=config.spinup.enabled,
        surface_forcing=config.surface_forcing,
    )


def _make_veros_component(state: Any, config: VerosConfig) -> Component:
    """Bind the existing owner to its current shared grid without reinitializing."""
    _veros_gcm_state, _veros_output, _veros_runtime, _ = _load_veros_implementation()
    component = CallableComponent(
        config.name,
        state.grid,
        partial(_veros_runtime.step_veros_runtime, state),
        spec=ComponentSpec(
            inputs=(
                tuple(_flatten_field_items(VERIS_TO_OCEAN_FLUX_FIELDS))
                if config.surface_forcing == "coupled"
                else (
                    _veros_gcm_state.VEROS_INPUT_FIELD_NAMES
                    if state.uses_atmosphere_forcing
                    else ()
                )
            ),
            outputs=(
                tuple(_flatten_field_items(OCEAN_TO_VERIS_FORCING_FIELDS))
                if config.surface_forcing == "coupled"
                else ("sea_surface_temperature",)
            ),
            initial_fields=_veros_gcm_state.veros_default_fields(
                coupled=config.surface_forcing == "coupled"
            ),
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
