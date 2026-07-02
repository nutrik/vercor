from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import jax
from jax.errors import JaxRuntimeError

from vercor.clock import Clock
from vercor.components.runtime_execution import host_component_names
from vercor.jax_logging import LoggerLike
from vercor.dtypes import as_jax_index_array
from vercor.runtime.dispatch_context import RuntimeDispatchContext
from vercor.runtime.driver import step_runtime_component
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.progress import (
    log_scanned_component_progress,
    log_scanned_step_progress,
    runtime_component_progress_message,
    runtime_step_progress_message,
    runtime_step_progress_messages,
)
from vercor.runtime.run_context import RuntimeRunContext
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.time import build_runtime_step_info, scalar_runtime_step_info
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray


def run_coupler_runtime(
    runtime_state: RuntimeCouplerState,
    *,
    context: RuntimeRunContext,
) -> RuntimeCouplerState:
    """Run a validated runtime state through the host or compiled scanned path."""

    with context.interrupts.signal_scope():
        host_names = host_component_names(context.dispatch_context.components)
        if not host_names:
            return _run_compiled_scanned_runtime(
                runtime_state,
                context=context,
            )
        _warn_non_differentiable_host_runtime(
            context.logger,
            host_names,
        )
        return run_host_runtime(
            runtime_state,
            run_sequence=context.run_sequence,
            clock=context.clock,
            settings=context.dispatch_context.settings,
            logger=context.logger,
            dispatch_context=context.dispatch_context,
            interrupts=context.interrupts,
        )


def _run_compiled_scanned_runtime(
    runtime_state: RuntimeCouplerState,
    *,
    context: RuntimeRunContext,
) -> RuntimeCouplerState:
    """Run a pure runtime state through a one-shot compiled scanned path."""

    try:

        def scanned_runtime(
            state: RuntimeCouplerState,
        ) -> RuntimeCouplerState:
            return run_scanned_runtime(
                state,
                run_sequence=context.run_sequence,
                clock=context.clock,
                settings=context.dispatch_context.settings,
                logger=context.logger,
                dispatch_context=context.dispatch_context,
                interrupts=context.interrupts,
            )

        compiled_runtime = cast(
            Callable[[RuntimeCouplerState], RuntimeCouplerState],
            jax.jit(scanned_runtime),
        )
        return compiled_runtime(runtime_state)
    except JaxRuntimeError as error:
        context.interrupts.raise_if_jax_callback_interrupted(
            error,
            "compiled scanned runtime",
        )


def _warn_non_differentiable_host_runtime(
    logger: LoggerLike,
    host_names: Sequence[str],
) -> None:
    """Warn that host-backed components make the coupled loop non-differentiable."""

    names = ", ".join(host_names)
    logger.warning(
        "Coupled loop is not differentiable because host-backed component(s) "
        f"require the Python runtime: {names}"
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
    step_progress_messages = runtime_step_progress_messages(clock)

    def step_all_components(
        state: RuntimeCouplerState,
        scan_input: tuple[RuntimeArray, Any],
    ) -> tuple[RuntimeCouplerState, None]:
        step_index, step_info = scan_input
        interrupts.scanned_checkpoint(
            "scanned runtime step",
            step_index,
        )
        log_scanned_step_progress(logger, step_index, step_progress_messages)
        for cname in run_sequence:
            interrupts.scanned_checkpoint(
                f"scanned runtime component {cname}",
                step_index,
            )
            log_scanned_component_progress(logger, cname)
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
