from datetime import datetime
from typing import Any, cast

import jax.numpy as jnp
import matplotlib.pyplot as plt

from vercor import Clock, RunSequence
from vercor.setups.coupler_helpers import (
    ExchangeSpec,
    add_exchange_specs,
    build_coupler,
)
from vercor.setups.slab.atmosphere import make_slab_atmosphere
from vercor.setups.slab.land import make_slab_land
from vercor.setups.slab.ocean import make_slab_ocean
from vercor.setups.slab.seaice import make_slab_seaice
from vercor.dtypes import jax_ones
from vercor.grid_geometry import make_rectilinear_grid
from vercor.regridders import bilinear, conservative
from vercor.setups.exchange_recipes import (
    LAND_TO_ATMOSPHERE_SOIL_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_SEAICE_SURFACE_FIELDS,
    SEAICE_TO_OCEAN_FIELDS,
    SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
    SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
)
from vercor.diagnostics import (
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
)

if __name__ == "__main__":
    # Build grids
    atm_grid = make_rectilinear_grid("atm-grid", 128, 64, 0.0, 360.0, -90.0, 90.0)

    ocn_grid_shape = (64, 32)
    binary_mask = jax_ones(ocn_grid_shape).T.at[:2, :].set(0.0)  # land points
    ocn_grid = make_rectilinear_grid(
        "ocn-grid", *ocn_grid_shape, 0.0, 360.0, -90.0, 90.0, mask=binary_mask
    )

    ice_grid = make_rectilinear_grid(
        "ice-grid", *ocn_grid_shape, 0.0, 360.0, -90.0, 90.0
    )
    lnd_grid = make_rectilinear_grid("lnd-grid", 128, 64, 0.0, 360.0, -90.0, 90.0)

    # Build components
    atm = make_slab_atmosphere(atm_grid)
    ocn = make_slab_ocean(ocn_grid)
    ice = make_slab_seaice(ice_grid)
    lnd = make_slab_land(lnd_grid)

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=3600, steps=24)
    run_sequence = RunSequence(order=["OCN", "ATM", "ICE", "LND"])

    # Coupler
    components: list[Any] = [atm, ocn, ice, lnd]
    cpl = build_coupler(
        clock=clock,
        components=components,
        run_sequence=run_sequence,
    )

    # Exchanges
    # scalar fields (vector field))
    # ["SHF", "LHF", ("u10m", "v10m")]
    add_exchange_specs(
        cpl,
        (
            ExchangeSpec(
                source="ATM",
                destination="OCN",
                field_names=SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="OCN",
                destination="ATM",
                field_names=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="OCN",
                destination="ICE",
                field_names=OCEAN_TO_SEAICE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="LND",
                destination="ATM",
                field_names=LAND_TO_ATMOSPHERE_SOIL_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="ATM",
                destination="LND",
                field_names=SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
                regridder_factory=conservative,
            ),
            ExchangeSpec(
                source="OCN",
                destination="ICE",
                field_names=OCEAN_TO_SEAICE_SURFACE_FIELDS,
                regridder_factory=conservative,
            ),
            ExchangeSpec(
                source="ICE",
                destination="OCN",
                field_names=SEAICE_TO_OCEAN_FIELDS,
                regridder_factory=conservative,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
    views = cpl.runtime_component_views(final_state, names=("ATM", "OCN", "LND", "ICE"))

    # Inspect a few fields in a component-wise table.
    print_component_field_means_table(
        components=views,
        fields=[
            ("sea_surface_temperature", "sst"),
            ("temperature_2m", "temperature_2m"),
            ("u_velocity_10m", "u_velocity_10m"),
            ("v_velocity_10m", "v_velocity_10m"),
            ("soil_moisture", "soil_moisture"),
            ("sensible_heat_flux", "sensible_heat_flux"),
        ],
        component_order=["ATM", "OCN", "LND"],
    )
    print(
        "ICE ice_fraction mean:",
        float(jnp.nanmean(jnp.asarray(views["ICE"].field("ice_fraction")))),
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            (
                "ATM",
                views["ATM"],
                "sea_surface_temperature",
                "u_velocity_10m",
                "v_velocity_10m",
            ),
            (
                "OCN",
                views["OCN"],
                "sea_surface_temperature",
                "u_velocity_10m",
                "v_velocity_10m",
            ),
        ],
        figsize=(15, 10),
        quiver_scale=100,
        cmap="coolwarm",
    )

    fig.colorbar(cast(Any, scalar_mappable), ax=axs, shrink=0.6)

    plt.show()
