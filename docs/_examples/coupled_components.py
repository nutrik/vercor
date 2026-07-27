"""Transfer a forcing field into a simple model component."""

from collections.abc import Mapping
from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, Exchange, RectilinearGrid
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    DataComponent,
    StepContext,
)
from vercor.regridding import bilinear
from vercor.types import RuntimeArray


def apply_heat_flux(
    fields: Mapping[str, RuntimeArray], context: StepContext
) -> Mapping[str, RuntimeArray]:
    """Apply the received forcing over one coupling time step."""

    _ = context
    return {"temperature": fields["temperature"] + fields["heat_flux"]}


grid = RectilinearGrid.uniform(
    "shared-grid",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
forcing = DataComponent("FORCING", grid, {"heat_flux": 5.0})
model = CallableComponent(
    "MODEL",
    grid,
    apply_heat_flux,
    spec=ComponentSpec(
        inputs=("heat_flux",),
        outputs=("temperature",),
        initial_fields={"heat_flux": 0.0, "temperature": 280.0},
    ),
)
clock = Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
coupler = Coupler(
    clock,
    components=(forcing, model),
    exchanges=(
        Exchange(
            "FORCING",
            "MODEL",
            ("heat_flux",),
            route_id="forcing-to-model",
            regridder_factory=bilinear,
        ),
    ),
    run_order=("FORCING", "MODEL"),
)

final_state = coupler.run(output=None)
temperature = final_state.component("MODEL").field("temperature")

assert bool(jnp.all(temperature == jnp.asarray(285.0)))
