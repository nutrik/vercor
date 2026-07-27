"""Run a host-backed component with an explicitly returned payload."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, RectilinearGrid, RuntimeOptions
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    LifecycleHooks,
    SetupContext,
    SetupResult,
    StepContext,
    StepResult,
)
from vercor.types import RuntimeArray


@dataclass(frozen=True)
class HostPayload:
    """Per-run host state returned by the lifecycle and step contracts."""

    calls: int = 0


def setup_host(component: object, context: SetupContext) -> SetupResult:
    """Create the first immutable host payload for one runtime state."""

    _ = component, context
    return SetupResult(payload=HostPayload())


def host_step(
    fields: Mapping[str, RuntimeArray],
    context: StepContext,
    payload: object | None,
) -> StepResult:
    """Use and replace payload state while evolving the declared field."""

    _ = context
    if not isinstance(payload, HostPayload):
        raise TypeError("host payload was not initialized")
    increment = payload.calls + 1
    return StepResult(
        fields={"counter": fields["counter"] + increment},
        payload=HostPayload(payload.calls + 1),
    )


grid = RectilinearGrid.uniform(
    "host-grid",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
host = CallableComponent(
    "HOST",
    grid,
    host_step,
    spec=ComponentSpec(
        outputs=("counter",),
        initial_fields={"counter": 0.0},
        execution="host",
        lifecycle=LifecycleHooks(setup=setup_host),
    ),
)
clock = Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=2)
coupler = Coupler(
    clock,
    components=(host,),
    run_order=("HOST",),
    runtime=RuntimeOptions(backend="auto"),
)
final_state = coupler.run(output=None)

assert bool(jnp.all(final_state.component("HOST").field("counter") == jnp.asarray(3.0)))
