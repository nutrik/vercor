"""Build static and monthly data components without an active model step."""

from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, RectilinearGrid
from vercor.components import ComponentSpec, DataComponent, TransferPolicy

grid = RectilinearGrid.uniform(
    "forcing-grid",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
static_forcing = DataComponent(
    "STATIC",
    grid,
    {"heat_flux": 25.0},
)
monthly_forcing = DataComponent(
    "MONTHLY",
    grid,
    {"temperature": jnp.arange(12.0)[:, None, None] * jnp.ones((12, *grid.shape))},
    spec=ComponentSpec(transfer=TransferPolicy("linear")),
)

clock = Clock(datetime(2000, 1, 1), dt_seconds=86_400.0, steps=0)
coupler = Coupler(clock, components=(static_forcing, monthly_forcing))
initial_state = coupler.initial_state()

assert initial_state.component("STATIC").field("heat_flux").shape == grid.shape
assert initial_state.component("MONTHLY").field("temperature").shape == (
    12,
    *grid.shape,
)
