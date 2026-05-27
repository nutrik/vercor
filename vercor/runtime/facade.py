from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import vercor.output as _output
from vercor.clock import Clock
from vercor.exchange import Exchange
from vercor.jax_logging import LoggerLike
from vercor.run_sequence import RunSequence
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.coupler_state import (
    refresh_runtime_contracts,
    runtime_state_from_components as _runtime_state_from_components,
    validate_runtime_state as _validate_runtime_state,
)
from vercor.runtime.dispatch_context import (
    RuntimeDispatchContext,
    build_runtime_dispatch_context,
)
from vercor.runtime.driver import prime_runtime_outgoing
from vercor.runtime.initialization import (
    RuntimeInitializationState,
    initialize_coupler_runtime as _initialize_coupler_runtime,
    validate_registered_component_setup as _validate_registered_component_setup,
)
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.run_context import RuntimeRunContext
from vercor.runtime.runner import (
    run_coupler_runtime,
    run_scanned_runtime,
)
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.time import initial_runtime_step_info
from vercor.runtime.topology import RuntimeRegridder
from vercor.runtime.views import RuntimeComponentView
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component


CompiledRuntimeCache = MutableMapping[
    tuple[Any, ...],
    Callable[[RuntimeCouplerState], RuntimeCouplerState],
]


@dataclass(frozen=True)
class PreparedRuntimeState:
    """Runtime state bundled with the refreshed contracts used to build it."""

    runtime_state: RuntimeCouplerState
    runtime_contracts: dict[str, RuntimeComponentContract]


def validate_registered_component_setup(component: "Component") -> None:
    """Validate one public component through the runtime setup boundary."""

    _validate_registered_component_setup(component)


def initialize_coupler_runtime(
    *,
    clock: Clock,
    components: dict[str, "Component"],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder],
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray],
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray],
    run_sequence: RunSequence,
    settings: VercorSettings,
    logger: LoggerLike,
    enable_x64_computations: bool | None = None,
) -> RuntimeInitializationState:
    """Initialize components, runtime contracts, and exchange topology."""

    return _initialize_coupler_runtime(
        clock=clock,
        components=components,
        exchanges=exchanges,
        regridders=regridders,
        binary_masks=binary_masks,
        fractional_masks=fractional_masks,
        run_sequence=run_sequence,
        settings=settings,
        logger=logger,
        enable_x64_computations=enable_x64_computations,
    )


def runtime_state_from_components(
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray],
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray],
    prefill_missing: bool,
) -> PreparedRuntimeState:
    """Build immutable runtime state from setup components and exchanges."""

    runtime_contracts = refresh_runtime_contracts(
        components,
        exchanges,
        validate_endpoints=False,
    )
    runtime_state = _runtime_state_from_components(
        components,
        exchanges,
        fractional_masks,
        binary_masks,
        contracts=runtime_contracts,
        prefill_missing=prefill_missing,
    )
    return PreparedRuntimeState(runtime_state, runtime_contracts)


def validate_runtime_state(
    runtime_state: RuntimeCouplerState,
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder],
    run_sequence: RunSequence,
) -> dict[str, RuntimeComponentContract]:
    """Validate runtime state and return the contracts used for validation."""

    runtime_contracts = refresh_runtime_contracts(
        components,
        exchanges,
        validate_endpoints=False,
    )
    _validate_runtime_state(
        runtime_state,
        components=components,
        exchanges=exchanges,
        regridders=regridders,
        contracts=runtime_contracts,
        run_sequence=tuple(run_sequence),
    )
    return runtime_contracts


def runtime_dispatch_context(
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder],
    runtime_contracts: Mapping[str, RuntimeComponentContract],
    clock: Clock,
    settings: VercorSettings,
) -> RuntimeDispatchContext:
    """Return static runtime dispatch plumbing for a configured coupler."""

    return build_runtime_dispatch_context(
        components,
        exchanges,
        regridders,
        runtime_contracts,
        dt_seconds=clock.dt_seconds,
        settings=settings,
    )


def create_runtime_state(
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder],
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray],
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray],
    run_sequence: RunSequence,
    clock: Clock,
    settings: VercorSettings,
    prefill_missing: bool,
) -> PreparedRuntimeState:
    """Create, prime, and validate immutable runtime state."""

    prepared = runtime_state_from_components(
        components=components,
        exchanges=exchanges,
        fractional_masks=fractional_masks,
        binary_masks=binary_masks,
        prefill_missing=prefill_missing,
    )
    runtime_state = prepared.runtime_state
    if prefill_missing and tuple(run_sequence):
        runtime_state = prime_runtime_outgoing(
            runtime_state,
            tuple(run_sequence),
            dispatch_context=runtime_dispatch_context(
                components=components,
                exchanges=exchanges,
                regridders=regridders,
                runtime_contracts=prepared.runtime_contracts,
                clock=clock,
                settings=settings,
            ),
            step_info=initial_runtime_step_info(clock, settings),
        )
    runtime_contracts = validate_runtime_state(
        runtime_state,
        components=components,
        exchanges=exchanges,
        regridders=regridders,
        run_sequence=run_sequence,
    )
    return PreparedRuntimeState(runtime_state, runtime_contracts)


def prepare_runtime_state(
    initial_state: RuntimeCouplerState | None,
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder],
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray],
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray],
    runtime_contracts: Mapping[str, RuntimeComponentContract],
    run_sequence: RunSequence,
    clock: Clock,
    settings: VercorSettings,
    validate_state: bool = True,
) -> PreparedRuntimeState:
    """Return a runtime state ready for execution."""

    if initial_state is None:
        return create_runtime_state(
            components=components,
            exchanges=exchanges,
            regridders=regridders,
            fractional_masks=fractional_masks,
            binary_masks=binary_masks,
            run_sequence=run_sequence,
            clock=clock,
            settings=settings,
            prefill_missing=True,
        )
    if validate_state:
        refreshed_contracts = validate_runtime_state(
            initial_state,
            components=components,
            exchanges=exchanges,
            regridders=regridders,
            run_sequence=run_sequence,
        )
        return PreparedRuntimeState(initial_state, refreshed_contracts)
    return PreparedRuntimeState(initial_state, dict(runtime_contracts))


def runtime_run_context(
    *,
    run_sequence: RunSequence,
    clock: Clock,
    logger: LoggerLike,
    log_level: int | str,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder],
    runtime_contracts: Mapping[str, RuntimeComponentContract],
    settings: VercorSettings,
    compiled_runtime_cache: CompiledRuntimeCache,
    interrupts: RuntimeInterruptController,
) -> RuntimeRunContext:
    """Return static runtime inputs bundled for execution."""

    return RuntimeRunContext(
        run_sequence=tuple(run_sequence),
        clock=clock,
        logger=logger,
        log_level=log_level,
        dispatch_context=runtime_dispatch_context(
            components=components,
            exchanges=exchanges,
            regridders=regridders,
            runtime_contracts=runtime_contracts,
            clock=clock,
            settings=settings,
        ),
        compiled_runtime_cache=compiled_runtime_cache,
        interrupts=interrupts,
    )


def run(
    runtime_state: RuntimeCouplerState,
    *,
    run_sequence: RunSequence,
    clock: Clock,
    logger: LoggerLike,
    log_level: int | str,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder],
    runtime_contracts: Mapping[str, RuntimeComponentContract],
    settings: VercorSettings,
    compiled_runtime_cache: CompiledRuntimeCache,
    interrupts: RuntimeInterruptController,
    donate_state: bool,
) -> RuntimeCouplerState:
    """Run a validated runtime state through the selected runtime path."""

    return run_coupler_runtime(
        runtime_state,
        context=runtime_run_context(
            run_sequence=run_sequence,
            clock=clock,
            logger=logger,
            log_level=log_level,
            components=components,
            exchanges=exchanges,
            regridders=regridders,
            runtime_contracts=runtime_contracts,
            settings=settings,
            compiled_runtime_cache=compiled_runtime_cache,
            interrupts=interrupts,
        ),
        donate_state=donate_state,
    )


def run_scanned(
    runtime_state: RuntimeCouplerState,
    *,
    run_sequence: RunSequence,
    clock: Clock,
    logger: LoggerLike,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder],
    runtime_contracts: Mapping[str, RuntimeComponentContract],
    settings: VercorSettings,
    interrupts: RuntimeInterruptController,
) -> RuntimeCouplerState:
    """Run the unified scanned runtime path and return final state."""

    return run_scanned_runtime(
        runtime_state,
        run_sequence=tuple(run_sequence),
        clock=clock,
        settings=settings,
        logger=logger,
        dispatch_context=runtime_dispatch_context(
            components=components,
            exchanges=exchanges,
            regridders=regridders,
            runtime_contracts=runtime_contracts,
            clock=clock,
            settings=settings,
        ),
        interrupts=interrupts,
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
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray],
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray],
    output_file_mask: Path | None,
    logger: LoggerLike,
) -> None:
    """Validate components and write final runtime output files."""

    for component in components.values():
        validate_registered_component_setup(component)
    _output.write_coupler_runtime_outputs(
        final_state=final_state,
        components=components,
        exchanges=exchanges,
        binary_masks=binary_masks,
        fractional_masks=fractional_masks,
        output_file_mask=output_file_mask,
        logger=logger,
    )


__all__ = [
    "PreparedRuntimeState",
    "create_runtime_state",
    "finalize",
    "initialize_coupler_runtime",
    "prepare_runtime_state",
    "run",
    "run_scanned",
    "runtime_component_view",
    "runtime_component_views",
    "runtime_dispatch_context",
    "runtime_run_context",
    "runtime_state_from_components",
    "validate_registered_component_setup",
    "validate_runtime_state",
]
