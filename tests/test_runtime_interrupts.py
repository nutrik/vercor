from __future__ import annotations

from datetime import datetime
import os
import signal
from typing import Any, cast

import jax
from jax.errors import JaxRuntimeError
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from tests._runtime_helpers import (
    clear_runtime_cache,
    compiled_runtime_cache_values,
    runtime_cache_entry_count,
)
from vercor.clock import Clock
from vercor.components.base import Component
from vercor.components.host import HostRuntimeComponent
from vercor.coupler import Coupler
from vercor.components.contexts import ComponentStepContext
from vercor.runtime.interrupts import (
    RuntimeInterruptController,
    RuntimeInterrupted,
    default_runtime_interrupt_signals,
)


class _NoopRuntimeComponent(Component):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, grid=make_test_grid(name=name.lower()))
        self.data["temperature"] = np.ones((2, 2), dtype=float)

    def step_runtime_state(
        self,
        component_state: Any,
        context: ComponentStepContext,
    ) -> Any:
        _ = context
        return component_state


class _InterruptingHostComponent(HostRuntimeComponent):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, grid=make_test_grid(name=name.lower()))
        self.data["temperature"] = np.ones((2, 2), dtype=float)

    def step_host_runtime_state(
        self,
        component_state: Any,
        context: ComponentStepContext,
    ) -> Any:
        _ = context
        signal.raise_signal(signal.SIGINT)
        return component_state


def _make_pure_coupler(steps: int = 2) -> Coupler:
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=steps)
    )
    coupler.components = {"ATM": cast(Any, _NoopRuntimeComponent("ATM"))}
    coupler.run_sequence = ("ATM",)
    return coupler


def _block_until_ready(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _write_wakeup_signal(
    controller: RuntimeInterruptController,
    signum: signal.Signals,
) -> None:
    wakeup = getattr(controller, "_wakeup")
    write_fd = getattr(wakeup, "_write_fd")
    assert write_fd is not None
    os.write(write_fd, bytes([int(signum)]))


def test_signal_scope_registers_and_restores_terminal_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {int(signum): object() for signum in selected_signals}
    installed_handlers: dict[int, Any] = {}
    calls: list[tuple[int, Any]] = []

    def fake_getsignal(signum: int) -> Any:
        return previous_handlers[int(signum)]

    def fake_signal(signum: int, handler: Any) -> Any:
        calls.append((int(signum), handler))
        installed_handlers[int(signum)] = handler
        return previous_handlers[int(signum)]

    monkeypatch.setattr(signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(signal, "signal", fake_signal)

    controller = RuntimeInterruptController(signals=selected_signals)

    with controller.signal_scope():
        assert callable(installed_handlers[int(signal.SIGINT)])
        assert callable(installed_handlers[int(signal.SIGTERM)])
        installed_handlers[int(signal.SIGTERM)](int(signal.SIGTERM), None)

        with pytest.raises(RuntimeInterrupted, match="SIGTERM") as excinfo:
            controller.checkpoint("unit test")

    assert isinstance(excinfo.value, KeyboardInterrupt)
    assert calls[-2:] == [
        (int(signal.SIGINT), previous_handlers[int(signal.SIGINT)]),
        (int(signal.SIGTERM), previous_handlers[int(signal.SIGTERM)]),
    ]
    assert controller.requested_signal is None


@pytest.mark.parametrize("signum", default_runtime_interrupt_signals())
def test_terminal_signals_request_runtime_interruption(
    signum: signal.Signals,
) -> None:
    controller = RuntimeInterruptController(signals=(signum,))
    controller.request_from_signal(int(signum), None)

    with pytest.raises(RuntimeInterrupted, match=signum.name) as excinfo:
        controller.checkpoint("runtime checkpoint")

    assert isinstance(excinfo.value, KeyboardInterrupt)
    assert excinfo.value.signum == int(signum)


def test_checkpoint_observes_wakeup_fd_signal_without_python_handler() -> None:
    controller = RuntimeInterruptController(signals=(signal.SIGINT,))

    with controller.signal_scope():
        _write_wakeup_signal(controller, signal.SIGTERM)
        controller.checkpoint("ignored wakeup signal")
        assert controller.requested_signal is None

        _write_wakeup_signal(controller, signal.SIGINT)
        with pytest.raises(RuntimeInterrupted, match="SIGINT") as excinfo:
            controller.checkpoint("wakeup fd checkpoint")

    assert excinfo.value.signum == int(signal.SIGINT)


def test_unrelated_jax_runtime_errors_are_preserved() -> None:
    controller = RuntimeInterruptController()
    error = JaxRuntimeError("unrelated runtime failure")
    callback_error = JaxRuntimeError(
        "INTERNAL: CpuCallback error calling callback: KeyboardInterrupt"
    )

    with pytest.raises(JaxRuntimeError, match="unrelated runtime failure") as excinfo:
        controller.raise_if_jax_callback_interrupted(error, "compiled runtime")

    assert excinfo.value is error

    with pytest.raises(JaxRuntimeError, match="KeyboardInterrupt") as callback_excinfo:
        controller.raise_if_jax_callback_interrupted(
            callback_error,
            "compiled runtime",
        )

    assert callback_excinfo.value is callback_error


def test_host_runtime_signal_aborts_through_shared_controller() -> None:
    coupler = Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1))
    coupler.components = {"ATM": cast(Any, _InterruptingHostComponent("ATM"))}
    coupler.run_sequence = ("ATM",)

    with pytest.raises(RuntimeInterrupted, match="SIGINT") as excinfo:
        coupler.run()

    assert excinfo.value.signum == int(signal.SIGINT)


def test_compiled_scanned_runtime_translates_interrupt_callback_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = _make_pure_coupler()
    interrupts = coupler._runtime_resources.interrupt_controller
    original_checkpoint = interrupts.checkpoint
    requested = False

    def request_once_then_checkpoint(label: str = "runtime") -> None:
        nonlocal requested
        if not requested:
            requested = True
            interrupts.request(signal.SIGINT)
        original_checkpoint(label)

    monkeypatch.setattr(
        interrupts,
        "checkpoint",
        request_once_then_checkpoint,
    )

    with pytest.raises(RuntimeInterrupted, match="SIGINT") as excinfo:
        coupler.run(donate_state=False)

    assert isinstance(excinfo.value.__cause__, JaxRuntimeError)


def test_compiled_scanned_runtime_observes_wakeup_fd_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = _make_pure_coupler()
    interrupts = coupler._runtime_resources.interrupt_controller
    original_checkpoint = interrupts.checkpoint
    injected = False

    def write_wakeup_once_then_checkpoint(label: str = "runtime") -> None:
        nonlocal injected
        if not injected:
            injected = True
            _write_wakeup_signal(interrupts, signal.SIGTSTP)
        original_checkpoint(label)

    monkeypatch.setattr(
        interrupts,
        "checkpoint",
        write_wakeup_once_then_checkpoint,
    )

    with pytest.raises(RuntimeInterrupted, match="SIGTSTP") as excinfo:
        coupler.run(donate_state=False)

    assert isinstance(excinfo.value.__cause__, JaxRuntimeError)


def test_interrupt_checkpoints_do_not_split_compiled_runtime_cache() -> None:
    coupler = _make_pure_coupler()
    clear_runtime_cache(coupler)

    first = _block_until_ready(coupler.run(donate_state=False))
    compiled = cast(Any, next(iter(compiled_runtime_cache_values(coupler))))
    first_cache_size = compiled._cache_size()
    second = _block_until_ready(coupler.run(donate_state=False))

    assert runtime_cache_entry_count(coupler) == 1
    assert compiled._cache_size() == first_cache_size
    assert first.component_names == ("ATM",)
    assert second.component_names == ("ATM",)
