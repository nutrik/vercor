"""Run a dependency-light bundled VerCOR component."""

from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, RectilinearGrid
from vercor.setups import make_slab_ocean

grid = RectilinearGrid.uniform(
    "quickstart",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
clock = Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=2)
ocean = make_slab_ocean(grid)
coupler = Coupler(clock, components=(ocean,), run_order=(ocean.name,))

final_state = coupler.run()
sea_surface_temperature = final_state.component(ocean.name).field(
    "sea_surface_temperature"
)

assert sea_surface_temperature.shape == grid.shape
assert bool(jnp.all(jnp.isfinite(sea_surface_temperature)))
