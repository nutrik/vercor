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
from vercor.runtime import facade as _runtime_facade
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.resources import CouplerRuntimeResources
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component
    import vercor.runtime.contracts as _runtime_contracts_module
    import vercor.runtime.dispatch_context as _runtime_dispatch_context_module
    import vercor.runtime.run_context as _runtime_run_context_module
    import vercor.runtime.state as _runtime_state_module
    import vercor.runtime.topology as _runtime_topology_module
    import vercor.runtime.views as _runtime_views_module


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
        _runtime_resources: runtime-owned holder for topology maps, runtime
            contracts, compiled runtime cache, and interrupt controller.
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
    _runtime_resources: CouplerRuntimeResources = field(
        default_factory=CouplerRuntimeResources,
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

    @property
    def _regridders(
        self,
    ) -> dict[tuple[str, str, str], _runtime_topology_module.RuntimeRegridder]:
        """Compatibility alias for runtime-owned exchange regridders."""

        return self._runtime_resources.regridders

    @_regridders.setter
    def _regridders(
        self,
        value: dict[tuple[str, str, str], _runtime_topology_module.RuntimeRegridder],
    ) -> None:
        self._runtime_resources.regridders = value

    @property
    def _binary_masks(self) -> dict[tuple[str, str, str], RuntimeArray]:
        """Compatibility alias for runtime-owned binary exchange masks."""

        return self._runtime_resources.binary_masks

    @_binary_masks.setter
    def _binary_masks(self, value: dict[tuple[str, str, str], RuntimeArray]) -> None:
        self._runtime_resources.binary_masks = value

    @property
    def _fractional_masks(self) -> dict[tuple[str, str, str], RuntimeArray]:
        """Compatibility alias for runtime-owned fractional exchange masks."""

        return self._runtime_resources.fractional_masks

    @_fractional_masks.setter
    def _fractional_masks(
        self, value: dict[tuple[str, str, str], RuntimeArray]
    ) -> None:
        self._runtime_resources.fractional_masks = value

    @property
    def _runtime_contracts(
        self,
    ) -> dict[str, _runtime_contracts_module.RuntimeComponentContract]:
        """Compatibility alias for runtime-owned component contracts."""

        return self._runtime_resources.contracts

    @_runtime_contracts.setter
    def _runtime_contracts(
        self,
        value: dict[str, _runtime_contracts_module.RuntimeComponentContract],
    ) -> None:
        self._runtime_resources.contracts = value

    @property
    def _compiled_runtime_cache(
        self,
    ) -> dict[
        tuple[Any, ...],
        Callable[
            [_runtime_state_module.RuntimeCouplerState],
            _runtime_state_module.RuntimeCouplerState,
        ],
    ]:
        """Compatibility alias for runtime-owned compiled runtime cache."""

        return self._runtime_resources.compiled_runtime_cache

    @_compiled_runtime_cache.setter
    def _compiled_runtime_cache(
        self,
        value: dict[
            tuple[Any, ...],
            Callable[
                [_runtime_state_module.RuntimeCouplerState],
                _runtime_state_module.RuntimeCouplerState,
            ],
        ],
    ) -> None:
        self._runtime_resources.compiled_runtime_cache = value

    @property
    def _runtime_interrupts(self) -> RuntimeInterruptController:
        """Compatibility alias for the runtime-owned interrupt controller."""

        return self._runtime_resources.interrupts

    @_runtime_interrupts.setter
    def _runtime_interrupts(self, value: RuntimeInterruptController) -> None:
        self._runtime_resources.interrupts = value

    def register(
        self,
        component: "Component",
    ) -> None:
        """
        Register a component with the coupler.

        Arguments:
            component: component instance to register
        """

        _runtime_facade.validate_registered_component_setup(component)
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

        initialized = _runtime_facade.initialize_coupler_runtime(
            clock=self.clock,
            components=self.components,
            exchanges=self.exchanges,
            run_sequence=self.run_sequence,
            settings=self.settings,
            logger=self.logger,
            runtime_resources=self._runtime_resources,
            enable_x64_computations=enable_x64_computations,
        )

        topology = initialized.topology
        self.ocn_fmask_on_atm_grid = topology.ocn_fmask_on_atm_grid
        self.lnd_fmask_on_atm_grid = topology.lnd_fmask_on_atm_grid
        self.lnd_bmask_on_atm_grid = topology.lnd_bmask_on_atm_grid

    def _runtime_state_from_components(
        self, *, prefill_missing: bool = False
    ) -> _runtime_state_module.RuntimeCouplerState:
        prepared = _runtime_facade.runtime_state_from_components(
            components=self.components,
            exchanges=self.exchanges,
            runtime_resources=self._runtime_resources,
            prefill_missing=prefill_missing,
        )
        return prepared.runtime_state

    def _validate_runtime_state(
        self,
        runtime_state: _runtime_state_module.RuntimeCouplerState,
    ) -> None:
        _runtime_facade.validate_runtime_state(
            runtime_state,
            components=self.components,
            exchanges=self.exchanges,
            runtime_resources=self._runtime_resources,
            run_sequence=self.run_sequence,
        )

    def _prepare_runtime_state(
        self,
        initial_state: _runtime_state_module.RuntimeCouplerState | None,
        *,
        validate_state: bool = True,
    ) -> _runtime_state_module.RuntimeCouplerState:
        """Return a runtime state ready for execution."""

        prepared = _runtime_facade.prepare_runtime_state(
            initial_state,
            components=self.components,
            exchanges=self.exchanges,
            runtime_resources=self._runtime_resources,
            run_sequence=self.run_sequence,
            clock=self.clock,
            settings=self.settings,
            validate_state=validate_state,
        )
        return prepared.runtime_state

    def create_runtime_state(
        self, *, prefill_missing: bool = True
    ) -> _runtime_state_module.RuntimeCouplerState:
        """Create and validate the immutable state used by the unified runtime."""

        prepared = _runtime_facade.create_runtime_state(
            components=self.components,
            exchanges=self.exchanges,
            runtime_resources=self._runtime_resources,
            run_sequence=self.run_sequence,
            clock=self.clock,
            settings=self.settings,
            prefill_missing=prefill_missing,
        )
        return prepared.runtime_state

    def _runtime_dispatch_context(
        self,
    ) -> _runtime_dispatch_context_module.RuntimeDispatchContext:
        """Return static runtime dispatch plumbing for the current coupler state."""

        return _runtime_facade.runtime_dispatch_context(
            components=self.components,
            exchanges=self.exchanges,
            runtime_resources=self._runtime_resources,
            clock=self.clock,
            settings=self.settings,
        )

    def _runtime_run_context(self) -> _runtime_run_context_module.RuntimeRunContext:
        """Return static runtime inputs bundled for execution."""

        return _runtime_facade.runtime_run_context(
            run_sequence=self.run_sequence,
            clock=self.clock,
            logger=self.logger,
            log_level=self.log_level,
            components=self.components,
            exchanges=self.exchanges,
            runtime_resources=self._runtime_resources,
            settings=self.settings,
        )

    def runtime_component_view(
        self,
        runtime_state: _runtime_state_module.RuntimeCouplerState,
        name: str,
    ) -> _runtime_views_module.RuntimeComponentView:
        """Return a single object containing component metadata and runtime fields."""

        return _runtime_facade.runtime_component_view(
            components=self.components,
            runtime_state=runtime_state,
            name=name,
        )

    def runtime_component_views(
        self,
        runtime_state: _runtime_state_module.RuntimeCouplerState,
        names: Sequence[str] | None = None,
    ) -> dict[str, _runtime_views_module.RuntimeComponentView]:
        """Return named runtime component views in component or requested order."""

        return _runtime_facade.runtime_component_views(
            components=self.components,
            runtime_state=runtime_state,
            names=names,
        )

    def finalize(
        self,
        final_state: _runtime_state_module.RuntimeCouplerState,
        output_file_mask: Optional[Path] = None,
    ) -> None:
        """
        Write final runtime component state to component output files.

        Arguments:
            final_state: runtime state returned by run/create_runtime_state
            output_file_mask: optional path mask for output files
        """

        self.logger.info(" ------------ Finalizing coupler and components ------------")
        _runtime_facade.finalize(
            final_state=final_state,
            components=self.components,
            exchanges=self.exchanges,
            runtime_resources=self._runtime_resources,
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
        initial_state: _runtime_state_module.RuntimeCouplerState | None = None,
        *,
        donate_state: bool = False,
    ) -> _runtime_state_module.RuntimeCouplerState:
        """
        Run all registered components through the unified runtime entrypoint.

        Pure differentiable components run through the cached JIT-scanned runtime.
        Host-backed components run through the Python host bridge. When
        ``donate_state`` is true for pure runs, callers must treat the input
        runtime state as consumed after this method returns.
        """

        runtime_state = self._prepare_runtime_state(initial_state)
        return _runtime_facade.run(
            runtime_state,
            run_sequence=self.run_sequence,
            clock=self.clock,
            logger=self.logger,
            log_level=self.log_level,
            components=self.components,
            exchanges=self.exchanges,
            runtime_resources=self._runtime_resources,
            settings=self.settings,
            donate_state=donate_state,
        )

    def _run_scanned_runtime(
        self,
        initial_state: _runtime_state_module.RuntimeCouplerState | None = None,
        *,
        validate_state: bool = True,
    ) -> _runtime_state_module.RuntimeCouplerState:
        """Run the unified scanned runtime path and return state."""

        runtime_state = self._prepare_runtime_state(
            initial_state,
            validate_state=validate_state,
        )
        return _runtime_facade.run_scanned(
            runtime_state,
            run_sequence=self.run_sequence,
            clock=self.clock,
            logger=self.logger,
            components=self.components,
            exchanges=self.exchanges,
            runtime_resources=self._runtime_resources,
            settings=self.settings,
        )
