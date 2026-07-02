from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import vercor.output as _output
from vercor.calendar import ModelDateTime
from vercor.clock import Clock
from vercor.exchange import Exchange
from vercor.jax_logging import LoggerLike
from vercor._run_order import normalize_run_sequence
from vercor.runtime.dispatch_context import (
    RuntimeDispatchContext,
    build_runtime_dispatch_context,
)
from vercor.runtime.initialization import (
    RuntimeInitializationState,
    initialize_coupler_runtime as _initialize_coupler_runtime,
)
from vercor.runtime.preparation import (
    create_runtime_state,
    prepare_runtime_state,
    runtime_state_from_components,
    validate_runtime_state,
)
from vercor.runtime.resources import CouplerRuntimeResources
from vercor.runtime.run_context import RuntimeRunContext
from vercor.runtime.runner import (
    run_coupler_runtime,
)
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.views import RuntimeComponentView
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.components.base import Component


@dataclass(frozen=True)
class RuntimeFacadeInputs:
    """Repeated static coupler inputs consumed by runtime facade helpers."""

    components: Mapping[str, "Component"]
    exchanges: Sequence[Exchange]
    runtime_resources: CouplerRuntimeResources
    run_sequence: Sequence[str]
    clock: Clock
    settings: VercorSettings


def initialize_coupler_runtime(
    *,
    inputs: RuntimeFacadeInputs,
    logger: LoggerLike,
    enable_x64_computations: bool | None = None,
) -> RuntimeInitializationState:
    """Initialize components, runtime contracts, and exchange topology."""

    initialized = _initialize_coupler_runtime(
        clock=inputs.clock,
        components=dict(inputs.components),
        exchanges=inputs.exchanges,
        topology_maps=inputs.runtime_resources.topology_maps,
        run_sequence=normalize_run_sequence(inputs.run_sequence),
        settings=inputs.settings,
        logger=logger,
        enable_x64_computations=enable_x64_computations,
    )
    inputs.runtime_resources.runtime_contracts = initialized.runtime_contracts
    inputs.runtime_resources.topology_maps = initialized.topology.topology_maps
    return initialized


def runtime_dispatch_context(
    *,
    inputs: RuntimeFacadeInputs,
) -> RuntimeDispatchContext:
    """Return static runtime dispatch plumbing for a configured coupler."""

    return build_runtime_dispatch_context(
        inputs.components,
        inputs.exchanges,
        inputs.runtime_resources.topology_maps.regridders,
        inputs.runtime_resources.runtime_contracts,
        dt_seconds=inputs.clock.dt_seconds,
        settings=inputs.settings,
    )


def runtime_run_context(
    *,
    inputs: RuntimeFacadeInputs,
    logger: LoggerLike,
) -> RuntimeRunContext:
    """Return static runtime inputs bundled for execution."""

    return RuntimeRunContext(
        run_sequence=tuple(normalize_run_sequence(inputs.run_sequence)),
        clock=inputs.clock,
        logger=logger,
        dispatch_context=runtime_dispatch_context(
            inputs=inputs,
        ),
        interrupts=inputs.runtime_resources.interrupt_controller,
    )


def run(
    runtime_state: RuntimeCouplerState,
    *,
    inputs: RuntimeFacadeInputs,
    logger: LoggerLike,
) -> RuntimeCouplerState:
    """Run a validated runtime state through the selected runtime path."""

    return run_coupler_runtime(
        runtime_state,
        context=runtime_run_context(
            inputs=inputs,
            logger=logger,
        ),
    )


def runtime_component_view(
    *,
    components: Mapping[str, "Component"],
    runtime_state: RuntimeCouplerState,
    name: str,
) -> RuntimeComponentView:
    """Return a single object containing component metadata and runtime fields."""

    return RuntimeComponentView.from_component_state(
        name,
        components[name].grid,
        runtime_state.get_component_state(name),
    )


def runtime_component_views(
    *,
    components: Mapping[str, "Component"],
    runtime_state: RuntimeCouplerState,
    names: Sequence[str] | None = None,
) -> dict[str, RuntimeComponentView]:
    """Return named runtime component views in component or requested order."""

    selected_names = tuple(components) if names is None else tuple(names)
    return {
        name: runtime_component_view(
            components=components,
            runtime_state=runtime_state,
            name=name,
        )
        for name in selected_names
    }


def finalize(
    *,
    final_state: RuntimeCouplerState,
    inputs: RuntimeFacadeInputs,
    output_file_mask: Path | None,
    logger: LoggerLike,
) -> None:
    """Write final runtime output files."""

    topology_maps = inputs.runtime_resources.topology_maps
    _output.write_coupler_runtime_outputs(
        final_state=final_state,
        components=inputs.components,
        exchanges=inputs.exchanges,
        binary_masks=topology_maps.binary_masks,
        fractional_masks=topology_maps.fractional_masks,
        output_file_mask=output_file_mask,
        logger=logger,
    )
    _output.write_coupler_component_snapshots(
        final_state=final_state,
        components=inputs.components,
        output_time=_final_snapshot_time(inputs.clock),
        logger=logger,
    )


def _final_snapshot_time(clock: Clock) -> datetime | ModelDateTime:
    """Return the model time represented by the final runtime state."""

    final_time: datetime | ModelDateTime = clock.start
    for _, time, _ in clock.iter():
        final_time = time
    return final_time


__all__ = [
    "RuntimeFacadeInputs",
    "create_runtime_state",
    "finalize",
    "initialize_coupler_runtime",
    "prepare_runtime_state",
    "run",
    "runtime_component_view",
    "runtime_component_views",
    "runtime_dispatch_context",
    "runtime_run_context",
    "runtime_state_from_components",
    "validate_runtime_state",
]
