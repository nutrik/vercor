from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from vercor.clock import Clock
from vercor.exceptions import CouplerError
from vercor.exchange import Exchange
from vercor.jax_logging import (
    JaxCallbackLogger,
    LoggerLike,
    configure_python_logger,
    setup_logger as _setup_logger,
)
from vercor.run_sequence import RunSequence
import vercor.output as _output
from vercor.runtime import (
    RuntimeComponentContract,
    RuntimeCouplerState,
)
from vercor.runtime.coupler_state import (
    refresh_runtime_contracts,
    runtime_state_from_components,
    validate_runtime_state as validate_coupler_runtime_state,
)
from vercor.runtime.dispatch_context import (
    RuntimeDispatchContext,
    build_runtime_dispatch_context,
)
from vercor.runtime.driver import prime_runtime_outgoing
from vercor.runtime.initialization import (
    initialize_coupler_runtime,
    validate_registered_component_setup,
)
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.runner import (
    run_coupler_runtime,
    run_scanned_runtime,
)
from vercor.runtime.run_context import RuntimeRunContext
from vercor.runtime.time import initial_runtime_step_info
from vercor.runtime.topology import RuntimeRegridder
from vercor.runtime.views import RuntimeComponentView
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component


@dataclass
class Coupler:
    """Public orchestration facade for configured component integrations.

    The coupler owns registration, exchange declarations, run sequence,
    regridder/mask setup, runtime-state creation, execution, and final output.
    The differentiable integration itself operates on immutable runtime state;
    component objects remain setup/configuration adapters rather than the
    traced integration state.

    Attributes:
        clock: Clock instance for managing simulation time
        log_level: logging threshold for coupler logs (e.g., "INFO", "DEBUG", etc.)
        logger: Logger instance for coupler logging
        run_sequence: sequence of component names defining the call (step) order
        components: mapping of component name to component instance
        exchanges: list of all Exchange instances
        settings: VercorSettings instance for coupler settings
        lnd_bmask_on_atm_grid: binary land mask regridded onto atmosphere grid
        ocn_fmask_on_atm_grid: fractional ocean mask regridded onto atmosphere grid
        lnd_fmask_on_atm_grid: fractional land mask regridded onto atmosphere grid
        _regridders: mapping of (source component name, destination component name)
                to Regridder instance (a pool of all available regridders)
        _binary_masks: mapping of (source component name, destination component name)
                to a binary mask array. This mask is used during regridding of fields
                to ensure that only valid (e.g., ocean or land) points are considered
                during the regridding process.
        _fractional_masks: mapping of (source component name, destination component name)
                to a fractional mask array. This mask is applied during field exchanges
                after regridding to ensure that only the appropriate portion from source
                grid cells of the forcing or boundary conditions is transferred to
                destination grid cells, reflecting the partial coverage of source grid cells
                within destination grid cells.
        _runtime_contracts: mapping of component name to RuntimeComponentContract instance
        _compiled_runtime_cache: mapping of static runtime topology keys to cached compiled runtime functions
        _runtime_interrupts: controller for signaling and handling runtime
            interrupts across host and JAX-traced runtime paths
    """

    clock: Clock
    log_level: int | str = "INFO"
    logger: LoggerLike = field(default_factory=_setup_logger)
    run_sequence: RunSequence = field(default_factory=RunSequence)
    components: dict[str, Component] = field(default_factory=dict)
    exchanges: list[Exchange] = field(default_factory=list)
    settings: VercorSettings = field(default_factory=VercorSettings)
    lnd_bmask_on_atm_grid: RuntimeArray = field(init=False)
    ocn_fmask_on_atm_grid: RuntimeArray = field(init=False)
    lnd_fmask_on_atm_grid: RuntimeArray = field(init=False)
    _regridders: dict[
        tuple[str, str, str],
        RuntimeRegridder,
    ] = field(default_factory=dict)
    _binary_masks: dict[tuple[str, str, str], RuntimeArray] = field(
        default_factory=dict
    )
    _fractional_masks: dict[tuple[str, str, str], RuntimeArray] = field(
        default_factory=dict
    )
    _runtime_contracts: dict[str, RuntimeComponentContract] = field(
        default_factory=dict
    )
    _compiled_runtime_cache: dict[
        tuple[Any, ...],
        Callable[[RuntimeCouplerState], RuntimeCouplerState],
    ] = field(default_factory=dict, init=False, repr=False)
    _runtime_interrupts: RuntimeInterruptController = field(
        default_factory=RuntimeInterruptController,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Apply the configured logging threshold at construction time."""

        if isinstance(self.logger, logging.Logger):
            self.logger = JaxCallbackLogger(
                configure_python_logger(self.logger, self.log_level)
            )
        elif isinstance(self.logger, JaxCallbackLogger):
            configure_python_logger(self.logger.logger, self.log_level)

        set_level = getattr(self.logger, "setLevel", None)
        if callable(set_level):
            set_level(self.log_level)

    def register(
        self,
        component: Component,
    ) -> None:
        """
        Register a component with the coupler.

        Arguments:
            component: component instance to register
        """

        validate_registered_component_setup(component)
        if component.name in self.components:
            raise CouplerError(f"Component {component.name} already registered")

        self.components[component.name] = component
        self.logger.info(f" Registered component {component.name}")

    def add_exchange(self, exchange: Exchange) -> None:
        """
        Add an exchange definition to the coupler.

        Arguments:
            exchange: Exchange instance defining the exchange between components to add
        """

        self.exchanges.append(exchange)
        formatted_field_names = ", ".join(
            ", ".join(item) if isinstance(item, tuple) else item
            for item in exchange.field_names
        )
        self.logger.info(
            f" Added exchange {exchange.name}: Fields ({formatted_field_names})"
        )

    def set_components_run_sequence(self, run_sequence: RunSequence) -> None:
        """
        Set the run sequence for the coupler components.

        Arguments:
            run_sequence: RunSequence instance defining the order of components execution
        """

        for cname in run_sequence:
            if cname not in self.components.keys():
                raise CouplerError(f"Component {cname} not registered in coupler")
        self.run_sequence = run_sequence
        self.logger.info(
            f" Set coupler components run sequence: {', '.join(self.run_sequence)}"
        )

    def initialize(self, enable_x64_computations: Optional[bool] = None) -> None:
        """
        Initialize the coupler and all registered components.
        """

        initialized = initialize_coupler_runtime(
            clock=self.clock,
            components=self.components,
            exchanges=self.exchanges,
            regridders=self._regridders,
            binary_masks=self._binary_masks,
            fractional_masks=self._fractional_masks,
            run_sequence=self.run_sequence,
            settings=self.settings,
            logger=self.logger,
            enable_x64_computations=enable_x64_computations,
        )

        self._runtime_contracts = initialized.runtime_contracts
        topology = initialized.topology
        self._regridders = topology.regridders
        self._binary_masks = topology.binary_masks
        self._fractional_masks = topology.fractional_masks
        self.ocn_fmask_on_atm_grid = topology.ocn_fmask_on_atm_grid
        self.lnd_fmask_on_atm_grid = topology.lnd_fmask_on_atm_grid
        self.lnd_bmask_on_atm_grid = topology.lnd_bmask_on_atm_grid

    def _runtime_state_from_components(
        self, *, prefill_missing: bool = False
    ) -> RuntimeCouplerState:
        self._runtime_contracts = refresh_runtime_contracts(
            self.components,
            self.exchanges,
            validate_endpoints=False,
        )
        return runtime_state_from_components(
            self.components,
            self.exchanges,
            self._fractional_masks,
            self._binary_masks,
            contracts=self._runtime_contracts,
            prefill_missing=prefill_missing,
        )

    def _validate_runtime_state(self, runtime_state: RuntimeCouplerState) -> None:
        self._runtime_contracts = refresh_runtime_contracts(
            self.components,
            self.exchanges,
            validate_endpoints=False,
        )

        validate_coupler_runtime_state(
            runtime_state,
            components=self.components,
            exchanges=self.exchanges,
            regridders=self._regridders,
            contracts=self._runtime_contracts,
            run_sequence=tuple(self.run_sequence),
        )

    def _prepare_runtime_state(
        self,
        initial_state: RuntimeCouplerState | None,
        *,
        validate_state: bool = True,
    ) -> RuntimeCouplerState:
        """Return a runtime state ready for execution."""

        runtime_state = (
            self.create_runtime_state(prefill_missing=True)
            if initial_state is None
            else initial_state
        )
        if validate_state:
            self._validate_runtime_state(runtime_state)
        return runtime_state

    def create_runtime_state(
        self, *, prefill_missing: bool = True
    ) -> RuntimeCouplerState:
        """Create and validate the immutable state used by the unified runtime."""

        runtime_state = self._runtime_state_from_components(
            prefill_missing=prefill_missing
        )
        if prefill_missing and tuple(self.run_sequence):
            runtime_state = prime_runtime_outgoing(
                runtime_state,
                tuple(self.run_sequence),
                dispatch_context=self._runtime_dispatch_context(),
                step_info=initial_runtime_step_info(self.clock, self.settings),
            )
        self._validate_runtime_state(runtime_state)
        return runtime_state

    def _runtime_dispatch_context(self) -> RuntimeDispatchContext:
        """Return static runtime dispatch plumbing for the current coupler state."""

        return build_runtime_dispatch_context(
            self.components,
            self.exchanges,
            self._regridders,
            self._runtime_contracts,
            dt_seconds=self.clock.dt_seconds,
            settings=self.settings,
        )

    def _runtime_run_context(self) -> RuntimeRunContext:
        """Return static runtime inputs bundled for execution."""

        dispatch_context = self._runtime_dispatch_context()
        return RuntimeRunContext(
            run_sequence=tuple(self.run_sequence),
            clock=self.clock,
            logger=self.logger,
            log_level=self.log_level,
            dispatch_context=dispatch_context,
            compiled_runtime_cache=self._compiled_runtime_cache,
            interrupts=self._runtime_interrupts,
        )

    def runtime_component_view(
        self,
        runtime_state: RuntimeCouplerState,
        name: str,
    ) -> RuntimeComponentView:
        """Return a single object containing component metadata and runtime fields."""

        return RuntimeComponentView.from_component_state(
            name,
            self.components[name].grid,
            runtime_state.get_component_state(name),
        )

    def runtime_component_views(
        self,
        runtime_state: RuntimeCouplerState,
        names: Sequence[str] | None = None,
    ) -> dict[str, RuntimeComponentView]:
        """Return named runtime component views in component or requested order."""

        selected_names = tuple(self.components) if names is None else tuple(names)
        return {
            name: self.runtime_component_view(runtime_state, name)
            for name in selected_names
        }

    def finalize(
        self,
        final_state: RuntimeCouplerState,
        output_file_mask: Optional[Path] = None,
    ) -> None:
        """
        Write final runtime component state to component output files.

        Arguments:
            final_state: runtime state returned by run/create_runtime_state
            output_file_mask: optional path mask for output files
        """

        self.logger.info(" ------------ Finalizing coupler and components ------------")
        for component in self.components.values():
            validate_registered_component_setup(component)
        _output.write_coupler_runtime_outputs(
            final_state=final_state,
            components=self.components,
            exchanges=self.exchanges,
            binary_masks=self._binary_masks,
            fractional_masks=self._fractional_masks,
            output_file_mask=output_file_mask,
            logger=self.logger,
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Run start: {self.clock.start}\n"
            f"├── Components: "
            + ", ".join(
                f"<{component.__class__.__name__}>({name})"
                for name, component in self.components.items()
            )
            + "\n"
            f"├── Exchanges: {', '.join(exchange.name for exchange in self.exchanges)}\n"
            f"└── Run sequence: {', '.join(self.run_sequence)}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(runstart={self.clock.start}, run_sequence={' -> '.join(self.run_sequence)})"

    def run(
        self,
        initial_state: RuntimeCouplerState | None = None,
        *,
        donate_state: bool = False,
    ) -> RuntimeCouplerState:
        """
        Run all registered components through the unified runtime entrypoint.

        Pure differentiable components run through the cached JIT-scanned runtime.
        Host-backed components run through the Python host bridge. When
        ``donate_state`` is true for pure runs, callers must treat the input
        runtime state as consumed after this method returns.
        """

        runtime_state = self._prepare_runtime_state(initial_state)
        return run_coupler_runtime(
            runtime_state,
            context=self._runtime_run_context(),
            donate_state=donate_state,
        )

    def _run_scanned_runtime(
        self,
        initial_state: RuntimeCouplerState | None = None,
        *,
        validate_state: bool = True,
    ) -> RuntimeCouplerState:
        """Run the unified scanned runtime path and return state."""

        runtime_state = self._prepare_runtime_state(
            initial_state,
            validate_state=validate_state,
        )
        return run_scanned_runtime(
            runtime_state,
            run_sequence=tuple(self.run_sequence),
            clock=self.clock,
            settings=self.settings,
            logger=self.logger,
            dispatch_context=self._runtime_dispatch_context(),
            interrupts=self._runtime_interrupts,
        )
