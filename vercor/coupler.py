"""Canonical public constructor-only coupler assembly and run facade."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from vercor.clock import Clock as _Clock
import vercor.components as _components
from vercor.components._adapter import normalize_component as _normalize_component
from vercor.exceptions import CouplerError as _CouplerError
from vercor.exchanges import Exchange as _Exchange
from vercor.jax_logging import (
    JaxCallbackLogger as _JaxCallbackLogger,
    LoggerLike as _LoggerLike,
    configure_python_logger as _configure_python_logger,
    normalize_log_level as _normalize_log_level,
    setup_logger as _setup_logger,
)
from vercor.physics import PhysicalConstants as _PhysicalConstants
from vercor.output import OutputTarget as _OutputTarget
from vercor._run_order import normalize_run_order as _normalize_run_order
from vercor._runtime.contracts import (
    validate_exchange_fan_in as _validate_exchange_fan_in,
)
import vercor._runtime.facade as _runtime_facade
from vercor._runtime.prepared import PreparedCoupling as _PreparedCoupling
from vercor.runtime import RuntimeOptions as _RuntimeOptions
from vercor.state import RunState as _RunState

__all__ = ["Coupler"]


def _materialize_configuration(values: object, *, label: str) -> tuple[Any, ...]:
    """Return one constructor collection as an owned immutable tuple."""

    if values is None or isinstance(values, (str, bytes)):
        raise _CouplerError(
            f"{label} must be an iterable, not {type(values).__name__}."
        )
    try:
        return tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise _CouplerError(f"{label} must be an iterable.") from exc


class Coupler:
    """Public constructor-only facade for one configured coupled integration.

    All components, exchanges, and the active run order are supplied together.
    Their collections are copied into immutable views; component author objects
    are retained and treated as immutable configuration. Runtime preparation is
    lazy and executes at most once for this fixed configuration.
    """

    def __init__(
        self,
        clock: _Clock,
        *,
        components: Iterable[_components.Component] = (),
        exchanges: Iterable[_Exchange] = (),
        run_order: Sequence[str] = (),
        runtime: _RuntimeOptions | None = None,
        constants: _PhysicalConstants | None = None,
        logger: _LoggerLike | None = None,
        log_level: int | str = "INFO",
    ) -> None:
        """Create a complete immutable coupling declaration."""

        if not isinstance(clock, _Clock):
            raise TypeError("clock must be Clock")
        if runtime is not None and not isinstance(runtime, _RuntimeOptions):
            raise TypeError("runtime must be RuntimeOptions or None")
        if constants is not None and not isinstance(constants, _PhysicalConstants):
            raise TypeError("constants must be PhysicalConstants or None")
        normalized_log_level = _normalize_log_level(log_level)
        if logger is not None and not isinstance(
            logger,
            (logging.Logger, _JaxCallbackLogger, _LoggerLike),
        ):
            raise TypeError("logger must satisfy LoggerLike or be None")

        if logger is None:
            configured_logger: _LoggerLike = _setup_logger(normalized_log_level)
        elif isinstance(logger, logging.Logger):
            configured_logger = _JaxCallbackLogger(
                _configure_python_logger(logger, normalized_log_level)
            )
        elif isinstance(logger, _JaxCallbackLogger):
            _configure_python_logger(logger.logger, normalized_log_level)
            configured_logger = logger
        else:
            logger.setLevel(normalized_log_level)
            configured_logger = logger
        configured_runtime = _RuntimeOptions() if runtime is None else runtime
        configured_constants = _PhysicalConstants() if constants is None else constants
        self._clock = clock
        self._log_level = log_level
        self._logger = configured_logger
        self._runtime = configured_runtime
        self._constants = configured_constants

        component_values = _materialize_configuration(
            components,
            label="components",
        )
        declarations: dict[str, Any] = {}
        public_components: dict[str, _components.Component] = {}
        for index, component in enumerate(component_values):
            try:
                declaration = _normalize_component(component)
            except _CouplerError as exc:
                raise _CouplerError(
                    f"components contains invalid component at index {index}: {exc}"
                ) from exc
            if declaration.name in declarations:
                raise _CouplerError(
                    f"Duplicate component name '{declaration.name}' in components."
                )
            declarations[declaration.name] = declaration
            public_components[declaration.name] = declaration.component
            self.logger.info(f"Registered component {declaration.name}")

        exchange_values = _materialize_configuration(exchanges, label="exchanges")
        normalized_exchanges: list[_Exchange] = []
        route_ids: set[str] = set()
        for index, exchange in enumerate(exchange_values):
            if not isinstance(exchange, _Exchange):
                raise _CouplerError(
                    "exchanges contains invalid exchange at index "
                    f"{index}: expected Exchange, got {type(exchange).__name__}."
                )
            if exchange.route_id in route_ids:
                raise _CouplerError(
                    f"Exchange route ID '{exchange.route_id}' must be unique; "
                    "provide explicit distinct route_id values for routes with "
                    "the same endpoints."
                )
            route_ids.add(exchange.route_id)
            if exchange.source not in declarations:
                raise _CouplerError(
                    f"Exchange '{exchange.route_id}' has unknown source component "
                    f"'{exchange.source}'."
                )
            if exchange.target not in declarations:
                raise _CouplerError(
                    f"Exchange '{exchange.route_id}' has unknown target component "
                    f"'{exchange.target}'."
                )
            normalized_exchanges.append(exchange)
            self.logger.info(f"Added exchange {exchange.route_id}")
        _validate_exchange_fan_in(normalized_exchanges)

        try:
            configured_run_order = _normalize_run_order(run_order)
        except TypeError as exc:
            raise _CouplerError(str(exc)) from exc
        duplicate_run_name = next(
            (
                name
                for name in configured_run_order
                if configured_run_order.count(name) > 1
            ),
            None,
        )
        if duplicate_run_name is not None:
            raise _CouplerError(
                f"Duplicate run-order component '{duplicate_run_name}'."
            )
        unknown_run_name = next(
            (name for name in configured_run_order if name not in declarations),
            None,
        )
        if unknown_run_name is not None:
            raise _CouplerError(f"Unknown run-order component '{unknown_run_name}'.")

        self._declarations = MappingProxyType(declarations)
        self._components_view = MappingProxyType(public_components)
        self._exchanges = tuple(normalized_exchanges)
        self._run_order = configured_run_order
        self._prepared: _PreparedCoupling | None = None
        if configured_run_order:
            self.logger.info(
                f"Set coupler components run order: {', '.join(configured_run_order)}"
            )

    @property
    def clock(self) -> _Clock:
        """Return the immutable clock configuration reference."""

        return self._clock

    @property
    def runtime(self) -> _RuntimeOptions:
        """Return the immutable runtime policy."""

        return self._runtime

    @property
    def constants(self) -> _PhysicalConstants:
        """Return the immutable physical constants."""

        return self._constants

    @property
    def logger(self) -> _LoggerLike:
        """Return the configured runtime logger."""

        return self._logger

    @property
    def log_level(self) -> int | str:
        """Return the configured logging threshold."""

        return self._log_level

    @property
    def components(self) -> Mapping[str, _components.Component]:
        """Return original public components in a stable immutable mapping."""

        return self._components_view

    @property
    def _runtime_components(self) -> Mapping[str, Any]:
        """Return normalized declarations for private runtime preparation."""

        return self._declarations

    @property
    def exchanges(self) -> tuple[_Exchange, ...]:
        """Return immutable exchange declarations."""

        return self._exchanges

    @property
    def run_order(self) -> tuple[str, ...]:
        """Return the immutable active component sequence."""

        return self._run_order

    def _ensure_prepared(self) -> _PreparedCoupling:
        """Return the one lazily prepared runtime boundary."""

        if self._prepared is None:
            self._prepared = _runtime_facade.prepare_coupling(
                components=self._runtime_components,
                exchanges=self.exchanges,
                run_order=self.run_order,
                clock=self.clock,
                constants=self.constants,
                runtime=self.runtime,
                logger=self.logger,
            )
        return self._prepared

    def initial_state(self, *, prefill_missing: bool = True) -> _RunState:
        """Create and validate the coupled runtime state."""

        return _runtime_facade.create_runtime_state(
            prepared=self._ensure_prepared(),
            prefill_missing=prefill_missing,
        )

    def run(
        self,
        state: _RunState | None = None,
        *,
        output: _OutputTarget | None = None,
    ) -> _RunState:
        """Run the configured workflow and optionally write selected outputs.

        ``output=None`` performs no I/O. A bare :class:`OutputTarget` enables
        period means, final fields, and registered snapshots; its flags disable
        each kind independently. Enabled I/O rejects traced state leaves, while
        output-free and all-disabled runs remain JIT- and gradient-compatible.
        """

        if output is not None and not isinstance(output, _OutputTarget):
            raise TypeError("output must be OutputTarget or None")

        prepared = self._ensure_prepared()
        runtime_state = _runtime_facade.prepare_runtime_state(
            state,
            prepared=prepared,
        )
        return _runtime_facade.run(
            runtime_state,
            prepared=prepared,
            logger=self.logger,
            output=output,
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Run start: {self.clock.start}\n"
            "├── Components: "
            + ", ".join(
                f"<{component.__class__.__name__}>({name})"
                for name, component in self.components.items()
            )
            + "\n"
            f"├── Exchanges: {', '.join(exchange.route_id for exchange in self.exchanges)}\n"
            f"└── Run order: {', '.join(self.run_order)}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(runstart={self.clock.start}, "
            f"run_order={' -> '.join(self.run_order)})"
        )


# Postponed annotations retain private import aliases in ``inspect.signature``.
# Publish the canonical runtime objects for the stable run contract.
Coupler.run.__annotations__.update(
    {
        "state": _RunState | None,
        "output": _OutputTarget | None,
        "return": _RunState,
    }
)
