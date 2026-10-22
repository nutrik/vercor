"""Keep native Veris compilation independent of host coupling intervals."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import numpy as np
import pytest

from tests._workflow_test_support import SequentialBackend
from tests.test_veris_component import cold_fields, small_coupler
from tests.test_veris_rollouts import native_payload
from vercor import Coupler
from vercor.components import CallableComponent, ComponentSpec, StepContext
from vercor.exceptions import ComponentError
from vercor.physics import PhysicalConstants

pytestmark = pytest.mark.fast_always


@pytest.mark.parametrize("backend", ["jax", "host", "auto", "custom"])
def test_native_veris_transition_traces_once_per_fixed_schema_run(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Host neighbors and custom drivers must reuse the native compiled step."""
    native = pytest.importorskip("veris")
    advance = native.advance
    traced: list[str] = []

    def observed_advance(*args: Any, **kwargs: Any) -> Any:
        traced.append(type(args[0].Area).__name__)
        return advance(*args, **kwargs)

    monkeypatch.setattr(native, "advance", observed_advance)
    base, ice, _, _ = small_coupler(steps=3)
    components = list(base.components.values())
    host_steps: list[int] = []
    if backend == "auto":

        def host_step(fields: Any, context: StepContext) -> dict[str, Any]:
            host_steps.append(int(context.step))
            assert context.time is not None
            return {}

        components.append(
            CallableComponent(
                "HOST", ice.grid, host_step, spec=ComponentSpec(execution="host")
            )
        )
    selected: Any = SequentialBackend() if backend == "custom" else backend
    coupler = Coupler(
        base.clock,
        components=components,
        exchanges=base.exchanges,
        run_order=tuple(component.name for component in components),
        runtime=replace(base.runtime, backend=selected),
        log_level="ERROR",
    )
    final = coupler.run()
    jax.block_until_ready(final)
    assert int(native_payload(final).checked_native_steps) == 6
    assert (
        len(traced) == 1
    ), "native transition was retraced during one fixed-schema run"
    if backend == "auto":
        assert host_steps == [0, 1, 2]


def test_compiled_native_step_keeps_constants_and_duration_dynamic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused executable must read current physical constants and duration."""
    native = pytest.importorskip("veris")
    advance = native.advance
    traces: list[str] = []

    def observed_advance(*args: Any, **kwargs: Any) -> Any:
        traces.append(type(args[0].Area).__name__)
        return advance(*args, **kwargs)

    monkeypatch.setattr(native, "advance", observed_advance)
    coupler, component, _, _ = small_coupler(steps=1)
    initial = native_payload(coupler.initial_state())
    owner = component.spec.lifecycle.setup.__self__
    context = StepContext(
        600.0, constants=coupler.constants, dtype=coupler.runtime.dtype
    )
    ordinary = owner.step(cold_fields(), context, initial)
    longer = owner.step(cold_fields(), replace(context, dt_seconds=1200.0), initial)
    changed = owner.step(
        cold_fields(),
        replace(
            context, constants=replace(PhysicalConstants(), freshwater_density=999.0)
        ),
        initial,
    )
    jax.block_until_ready((ordinary, longer, changed))
    assert np.max(np.abs(ordinary.fields["sea_surface_heat_flux"])) > 0
    np.testing.assert_allclose(
        longer.fields["sea_surface_heat_flux"],
        np.asarray(ordinary.fields["sea_surface_heat_flux"]) / 2,
    )
    assert float(changed.payload.physical_constants.rhoFresh) == 999.0
    assert len(traces) == 1
    with pytest.raises(ComponentError, match="initialized native runtime payload"):
        owner.step(cold_fields(), context, None)
