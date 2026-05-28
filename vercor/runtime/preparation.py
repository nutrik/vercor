from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vercor.clock import Clock
from vercor.exchange import Exchange
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.coupler_state import (
    refresh_runtime_contracts,
    runtime_state_from_components as _runtime_state_from_components,
    validate_runtime_state as _validate_runtime_state,
)
from vercor.runtime.dispatch_context import build_runtime_dispatch_context
from vercor.runtime.driver import prime_runtime_outgoing
from vercor.runtime.resources import CouplerRuntimeResources
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.time import initial_runtime_step_info
from vercor.run_sequence import RunSequence
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.components.base import Component


class RuntimePreparationInputs(Protocol):
    """Static coupler inputs required to prepare immutable runtime state."""

    @property
    def components(self) -> Mapping[str, "Component"]:
        """Return configured setup components by name."""

    @property
    def exchanges(self) -> Sequence[Exchange]:
        """Return configured exchange declarations."""

    @property
    def runtime_resources(self) -> CouplerRuntimeResources:
        """Return mutable runtime resources for this coupler."""

    @property
    def run_sequence(self) -> RunSequence:
        """Return the configured runtime component order."""

    @property
    def clock(self) -> Clock:
        """Return the configured coupler clock."""

    @property
    def settings(self) -> VercorSettings:
        """Return the configured coupler settings."""


@dataclass(frozen=True)
class PreparedRuntimeState:
    """Runtime state bundled with the refreshed contracts used to build it."""

    runtime_state: RuntimeCouplerState
    runtime_contracts: dict[str, RuntimeComponentContract]


def runtime_state_from_components(
    *,
    inputs: RuntimePreparationInputs,
    prefill_missing: bool,
) -> PreparedRuntimeState:
    """Build immutable runtime state from setup components and exchanges."""

    runtime_contracts = refresh_runtime_contracts(
        inputs.components,
        inputs.exchanges,
        validate_endpoints=False,
    )
    inputs.runtime_resources.replace_contracts(runtime_contracts)
    runtime_state = _runtime_state_from_components(
        inputs.components,
        inputs.exchanges,
        inputs.runtime_resources.fractional_masks,
        inputs.runtime_resources.binary_masks,
        contracts=inputs.runtime_resources.contracts,
        prefill_missing=prefill_missing,
    )
    return PreparedRuntimeState(runtime_state, inputs.runtime_resources.contracts)


def validate_runtime_state(
    runtime_state: RuntimeCouplerState,
    *,
    inputs: RuntimePreparationInputs,
) -> dict[str, RuntimeComponentContract]:
    """Validate runtime state and return the contracts used for validation."""

    runtime_contracts = refresh_runtime_contracts(
        inputs.components,
        inputs.exchanges,
        validate_endpoints=False,
    )
    inputs.runtime_resources.replace_contracts(runtime_contracts)
    _validate_runtime_state(
        runtime_state,
        components=inputs.components,
        exchanges=inputs.exchanges,
        regridders=inputs.runtime_resources.regridders,
        contracts=inputs.runtime_resources.contracts,
        run_sequence=tuple(inputs.run_sequence),
    )
    return inputs.runtime_resources.contracts


def create_runtime_state(
    *,
    inputs: RuntimePreparationInputs,
    prefill_missing: bool,
) -> PreparedRuntimeState:
    """Create, prime, and validate immutable runtime state."""

    prepared = runtime_state_from_components(
        inputs=inputs,
        prefill_missing=prefill_missing,
    )
    runtime_state = prepared.runtime_state
    if prefill_missing and tuple(inputs.run_sequence):
        dispatch_context = build_runtime_dispatch_context(
            inputs.components,
            inputs.exchanges,
            inputs.runtime_resources.regridders,
            inputs.runtime_resources.contracts,
            dt_seconds=inputs.clock.dt_seconds,
            settings=inputs.settings,
        )
        runtime_state = prime_runtime_outgoing(
            runtime_state,
            tuple(inputs.run_sequence),
            dispatch_context=dispatch_context,
            step_info=initial_runtime_step_info(inputs.clock, inputs.settings),
        )
    runtime_contracts = validate_runtime_state(
        runtime_state,
        inputs=inputs,
    )
    return PreparedRuntimeState(runtime_state, runtime_contracts)


def prepare_runtime_state(
    initial_state: RuntimeCouplerState | None,
    *,
    inputs: RuntimePreparationInputs,
    validate_state: bool = True,
) -> PreparedRuntimeState:
    """Return a runtime state ready for execution."""

    if initial_state is None:
        return create_runtime_state(
            inputs=inputs,
            prefill_missing=True,
        )
    if validate_state:
        refreshed_contracts = validate_runtime_state(
            initial_state,
            inputs=inputs,
        )
        return PreparedRuntimeState(initial_state, refreshed_contracts)
    return PreparedRuntimeState(
        initial_state,
        dict(inputs.runtime_resources.contracts),
    )


__all__ = [
    "PreparedRuntimeState",
    "RuntimePreparationInputs",
    "create_runtime_state",
    "prepare_runtime_state",
    "runtime_state_from_components",
    "validate_runtime_state",
]
