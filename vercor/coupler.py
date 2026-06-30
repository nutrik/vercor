from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from vercor.clock import Clock
from vercor.exceptions import CouplerError
from vercor.exchange import Exchange
from vercor.jax_logging import (
    JaxCallbackLogger,
    LoggerLike,
    configure_python_logger,
    setup_logger as _setup_logger,
)
from vercor.run_sequence import normalize_run_sequence
import vercor.runtime.facade as _runtime_facade
from vercor.runtime.resources import CouplerRuntimeResources
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component
    import vercor.runtime.state as _runtime_state_module
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
    run_sequence: Sequence[str] = field(default_factory=tuple)
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

        self.run_sequence = normalize_run_sequence(self.run_sequence)

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

    def set_components_run_sequence(
        self,
        run_sequence: Sequence[str],
    ) -> None:
        """
        Set the run sequence for the coupler components.

        Arguments:
            run_sequence: component names defining the order of components execution
        """

        normalized_run_sequence = normalize_run_sequence(run_sequence)
        for cname in normalized_run_sequence:
            if cname not in self.components.keys():
                raise CouplerError(f"Component {cname} not registered in coupler")
        self.run_sequence = normalized_run_sequence
        self.logger.info(
            f" Set coupler components run sequence: {', '.join(self.run_sequence)}"
        )

    def _runtime_inputs(self) -> _runtime_facade.RuntimeFacadeInputs:
        """Return the repeated runtime facade input bundle for this coupler."""

        self.run_sequence = normalize_run_sequence(self.run_sequence)
        return _runtime_facade.RuntimeFacadeInputs(
            self.components,
            self.exchanges,
            self._runtime_resources,
            self.run_sequence,
            self.clock,
            self.settings,
        )

    def initialize(self, enable_x64_computations: Optional[bool] = None) -> None:
        """
        Initialize the coupler and all registered components.
        """

        initialized = _runtime_facade.initialize_coupler_runtime(
            inputs=self._runtime_inputs(),
            logger=self.logger,
            enable_x64_computations=enable_x64_computations,
        )

        topology = initialized.topology
        surface_masks = topology.surface_masks
        self.ocn_fmask_on_atm_grid = surface_masks.ocn_fmask_on_atm_grid
        self.lnd_fmask_on_atm_grid = surface_masks.lnd_fmask_on_atm_grid
        self.lnd_bmask_on_atm_grid = surface_masks.lnd_bmask_on_atm_grid

    def create_runtime_state(
        self, *, prefill_missing: bool = True
    ) -> _runtime_state_module.RuntimeCouplerState:
        """Create and validate the immutable state used by the unified runtime."""

        prepared = _runtime_facade.create_runtime_state(
            inputs=self._runtime_inputs(),
            prefill_missing=prefill_missing,
        )
        return prepared.runtime_state

    def clear_runtime_cache(self) -> None:
        """Clear compiled runtime entries cached for this coupler instance."""

        self._runtime_resources.runtime_cache.clear()

    def runtime_cache_entry_count(self) -> int:
        """Return the number of compiled runtime entries cached by this coupler."""

        return self._runtime_resources.runtime_cache.entry_count()

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
            inputs=self._runtime_inputs(),
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

        prepared = _runtime_facade.prepare_runtime_state(
            initial_state,
            inputs=self._runtime_inputs(),
        )
        return _runtime_facade.run(
            prepared.runtime_state,
            inputs=self._runtime_inputs(),
            logger=self.logger,
            log_level=self.log_level,
            donate_state=donate_state,
        )
