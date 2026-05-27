from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import logging
from typing import TYPE_CHECKING, Any, cast

import jax
from jax.errors import JaxRuntimeError

from vercor.clock import Clock
from vercor.exceptions import CouplerError
from vercor.exchange import Exchange
from vercor.jax_logging import (
    LoggerLike,
    effective_log_level,
    emit_host_log,
    logger_enabled_for,
)
from vercor.dtypes import as_jax_index_array
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.driver import (
    RuntimeDispatchContext,
    host_component_names,
    step_runtime_component,
)
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.time import build_runtime_step_info, scalar_runtime_step_info
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component

CompiledRuntime = Callable[[RuntimeCouplerState], RuntimeCouplerState]


def runtime_step_progress_message(n: int, time: object, dt: object) -> str:
    """Return the shared host/scanned runtime step progress message."""

    return f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} "


def runtime_component_progress_message(component_name: str) -> str:
    """Return the shared host/scanned runtime component progress message."""

    return f" Run component: {component_name}"


def run_coupler_runtime(
    runtime_state: RuntimeCouplerState,
    *,
    components: Mapping[str, Component],
    run_sequence: Sequence[str],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
    contracts: Mapping[str, RuntimeComponentContract],
    clock: Clock,
    settings: VercorSettings,
    logger: LoggerLike,
    log_level: int | str,
    dispatch_context: RuntimeDispatchContext,
    compiled_runtime_cache: MutableMapping[tuple[Any, ...], CompiledRuntime],
    interrupts: RuntimeInterruptController,
    donate_state: bool,
) -> RuntimeCouplerState:
    """Run a validated runtime state through the host or compiled scanned path."""

    with interrupts.signal_scope():
        host_names = host_component_names(components)
        if not host_names:
            try:
                cache_key = compiled_runtime_cache_key(
                    donate_state=donate_state,
                    components=components,
                    run_sequence=run_sequence,
                    exchanges=exchanges,
                    regridders=regridders,
                    logger=logger,
                    interrupts=interrupts,
                    log_level=log_level,
                    contracts=contracts,
                    clock=clock,
                    settings=settings,
                )

                def scanned_runtime(
                    state: RuntimeCouplerState,
                ) -> RuntimeCouplerState:
                    return run_scanned_runtime(
                        state,
                        run_sequence=run_sequence,
                        clock=clock,
                        settings=settings,
                        logger=logger,
                        dispatch_context=dispatch_context,
                        interrupts=interrupts,
                    )

                return compiled_scanned_runtime(
                    scanned_runtime,
                    cache=compiled_runtime_cache,
                    cache_key=cache_key,
                    donate_state=donate_state,
                )(runtime_state)
            except JaxRuntimeError as error:
                interrupts.raise_if_jax_callback_interrupted(
                    error,
                    "compiled scanned runtime",
                )

        if donate_state:
            names = ", ".join(host_names)
            raise CouplerError(
                "Runtime state donation is only supported for differentiable "
                f"components; host-backed component(s) require non-donating run(): {names}"
            )

        return run_host_runtime(
            runtime_state,
            run_sequence=run_sequence,
            clock=clock,
            settings=settings,
            logger=logger,
            dispatch_context=dispatch_context,
            interrupts=interrupts,
        )


def run_host_runtime(
    runtime_state: RuntimeCouplerState,
    *,
    run_sequence: Sequence[str],
    clock: Clock,
    settings: VercorSettings,
    logger: LoggerLike,
    dispatch_context: RuntimeDispatchContext,
    interrupts: RuntimeInterruptController,
) -> RuntimeCouplerState:
    """Run the host-enabled runtime path for non-differentiable adapters."""

    for n, time, dt in clock.iter():
        interrupts.checkpoint("host runtime step")
        logger.info(runtime_step_progress_message(n, time, dt))
        step_info = scalar_runtime_step_info(time, clock, settings)

        for cname in run_sequence:
            interrupts.checkpoint(f"host runtime component {cname}")
            logger.info(runtime_component_progress_message(cname))
            runtime_state = step_runtime_component(
                runtime_state,
                cname,
                step_info,
                dispatch_context=dispatch_context,
                allow_host_runtime=True,
                time=time,
                logger=logger,
            )
            interrupts.checkpoint(f"host runtime component {cname}")
        interrupts.checkpoint("host runtime step")

    return runtime_state


def run_scanned_runtime(
    runtime_state: RuntimeCouplerState,
    *,
    run_sequence: Sequence[str],
    clock: Clock,
    settings: VercorSettings,
    logger: LoggerLike,
    dispatch_context: RuntimeDispatchContext,
    interrupts: RuntimeInterruptController,
) -> RuntimeCouplerState:
    """Run the unified runtime path under ``jax.lax.scan`` and return state."""

    step_infos = build_runtime_step_info(clock, settings)
    step_indices = as_jax_index_array(range(clock.steps))
    step_progress_messages = tuple(
        runtime_step_progress_message(n, time, dt) for n, time, dt in clock.iter()
    )

    def log_scanned_step_progress(step_index: RuntimeArray) -> None:
        if not logger_enabled_for(logger, logging.INFO):
            return

        def emit(index: RuntimeArray) -> None:
            host_index = int(jax.device_get(index).item())
            emit_host_log(
                logger,
                logging.INFO,
                step_progress_messages[host_index],
            )

        jax.debug.callback(emit, step_index, ordered=True)

    def log_scanned_component_progress(component_name: str) -> None:
        if not logger_enabled_for(logger, logging.INFO):
            return

        jax.debug.callback(
            lambda: emit_host_log(
                logger,
                logging.INFO,
                runtime_component_progress_message(component_name),
            ),
            ordered=True,
        )

    def step_all_components(
        state: RuntimeCouplerState,
        scan_input: tuple[RuntimeArray, Any],
    ) -> tuple[RuntimeCouplerState, None]:
        step_index, step_info = scan_input
        interrupts.scanned_checkpoint(
            "scanned runtime step",
            step_index,
        )
        log_scanned_step_progress(step_index)
        for cname in run_sequence:
            interrupts.scanned_checkpoint(
                f"scanned runtime component {cname}",
                step_index,
            )
            log_scanned_component_progress(cname)
            state = step_runtime_component(
                state,
                cname,
                step_info,
                dispatch_context=dispatch_context,
                allow_host_runtime=False,
                logger=logger,
            )
            interrupts.scanned_checkpoint(
                f"scanned runtime component {cname}",
                step_index,
            )
        interrupts.scanned_checkpoint(
            "scanned runtime step",
            step_index,
        )
        return state, None

    try:
        final_state, _ = jax.lax.scan(
            step_all_components,
            runtime_state,
            (step_indices, step_infos),
            length=clock.steps,
        )
    except JaxRuntimeError as error:
        interrupts.raise_if_jax_callback_interrupted(
            error,
            "scanned runtime",
        )
    return final_state


def compiled_scanned_runtime(
    scanned_runtime: CompiledRuntime,
    *,
    cache: MutableMapping[tuple[Any, ...], CompiledRuntime],
    cache_key: tuple[Any, ...],
    donate_state: bool,
) -> CompiledRuntime:
    """Return a cached JIT-scanned runtime for one static topology key."""

    if cache_key in cache:
        return cache[cache_key]

    if donate_state:
        compiled_runtime = cast(
            CompiledRuntime,
            jax.jit(scanned_runtime, donate_argnums=(0,)),
        )
    else:
        compiled_runtime = cast(
            CompiledRuntime,
            jax.jit(scanned_runtime),
        )
    cache[cache_key] = compiled_runtime
    return compiled_runtime


def compiled_runtime_cache_key(
    *,
    donate_state: bool,
    components: Mapping[str, Component],
    run_sequence: Sequence[str],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
    logger: LoggerLike,
    interrupts: RuntimeInterruptController,
    log_level: int | str,
    contracts: Mapping[str, RuntimeComponentContract],
    clock: Clock,
    settings: VercorSettings,
) -> tuple[Any, ...]:
    """Return a static cache key for the compiled pure-runtime wrapper."""

    return (
        donate_state,
        tuple((name, id(component)) for name, component in components.items()),
        tuple(run_sequence),
        tuple(
            (
                id(exchange),
                exchange.source,
                exchange.destination,
                exchange.interpolation_type,
                tuple(exchange.field_names),
            )
            for exchange in exchanges
        ),
        tuple(sorted((key, id(value)) for key, value in regridders.items())),
        id(logger),
        id(interrupts),
        effective_log_level(logger, log_level),
        tuple(
            (name, contract.imports, contract.exports)
            for name, contract in sorted(contracts.items())
        ),
        repr(clock.start),
        clock.dt_seconds,
        clock.steps,
        clock.year_type,
        settings.year_in_seconds,
    )
