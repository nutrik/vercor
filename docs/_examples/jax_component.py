"""Differentiate and compile a pure JAX-backed component."""

from collections.abc import Mapping
from datetime import datetime

import jax
import jax.numpy as jnp

from vercor import Clock, Coupler, RectilinearGrid, RuntimeOptions
from vercor.components import CallableComponent, ComponentSpec, StepContext
from vercor.types import RuntimeArray


def heat_temperature(
    fields: Mapping[str, RuntimeArray], context: StepContext
) -> Mapping[str, RuntimeArray]:
    """Apply a constant heating tendency using only array operations."""

    return {
        "temperature": fields["temperature"]
        + fields["heating_rate"] * context.dt_seconds
    }


grid = RectilinearGrid.uniform(
    "jax-grid",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
model = CallableComponent(
    "MODEL",
    grid,
    heat_temperature,
    spec=ComponentSpec(
        inputs=("temperature", "heating_rate"),
        outputs=("temperature",),
        initial_fields={"temperature": 280.0, "heating_rate": 0.001},
        execution="jax",
    ),
)
clock = Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=2)
coupler = Coupler(
    clock,
    components=(model,),
    run_order=("MODEL",),
    runtime=RuntimeOptions(backend="jax"),
)
initial_state = coupler.initial_state()


def final_temperature_sum(initial_temperature: jax.Array) -> jax.Array:
    """Return the final temperature sum for a replacement initial condition."""

    state = initial_state.replace_fields(
        "MODEL",
        {"temperature": jnp.full(grid.shape, initial_temperature)},
    )
    result = coupler.run(state, output=None)
    return jnp.sum(result.component("MODEL").field("temperature"))


gradient = jax.grad(final_temperature_sum)(jnp.asarray(280.0))
jitted_sum = jax.jit(final_temperature_sum)(jnp.asarray(280.0))

assert bool(gradient == grid.shape[0] * grid.shape[1])
assert bool(jnp.isfinite(jitted_sum))
