from datetime import datetime
from typing import Any, cast

import jax.numpy as jnp
import matplotlib.pyplot as plt

from vercor import Clock, Exchange, RunSequence
from vercor.setups.coupler_helpers import add_exchanges, build_coupler
from vercor.setups.slab.atmosphere import make_slab_atmosphere
from vercor.setups.slab.land import make_slab_land
from vercor.setups.slab.ocean import make_slab_ocean
from vercor.setups.slab.seaice import make_slab_seaice
from vercor.dtypes import jax_ones
from vercor.grid_geometry import make_rectilinear_grid
from vercor.regridders import bilinear, conservative
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
    add_exchanges(
        cpl,
        (
            Exchange(
                source="ATM",
                destination="OCN",
                field_names=[
                    ("u_velocity_10m", "v_velocity_10m"),
                    "sensible_heat_flux",
                    "latent_heat_flux",
                ],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="OCN",
                destination="ATM",
                field_names=["sea_surface_temperature"],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="OCN",
                destination="ICE",
                field_names=["sea_surface_temperature"],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="LND",
                destination="ATM",
                field_names=["soil_moisture"],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="ATM",
                destination="LND",
                field_names=["latent_heat_flux", "sensible_heat_flux"],
                regridder_factory=conservative,
            ),
            Exchange(
                source="OCN",
                destination="ICE",
                field_names=["sea_surface_temperature"],
                regridder_factory=conservative,
            ),
            Exchange(
                source="ICE",
                destination="OCN",
                field_names=["ice_fraction"],
                regridder_factory=conservative,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)

    # Inspect a few fields in a component-wise table.
    print_component_field_means_table(
        components={
            "ATM": cpl.runtime_component_view(final_state, "ATM"),
            "OCN": cpl.runtime_component_view(final_state, "OCN"),
            "LND": cpl.runtime_component_view(final_state, "LND"),
        },
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
        float(
            jnp.nanmean(
                jnp.asarray(
                    final_state.get_component_state("ICE").data.get("ice_fraction")
                )
            )
        ),
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            (
                "ATM",
                cpl.runtime_component_view(final_state, "ATM"),
                "sea_surface_temperature",
                "u_velocity_10m",
                "v_velocity_10m",
            ),
            (
                "OCN",
                cpl.runtime_component_view(final_state, "OCN"),
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
