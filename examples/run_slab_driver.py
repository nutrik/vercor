from datetime import datetime
from typing import Any, cast

import jax.numpy as jnp
import matplotlib.pyplot as plt

from vercor import Clock, Coupler, Exchange
from vercor.dtypes import jax_ones
from vercor.grids import rectilinear
from vercor.regridding import bilinear, conservative
from vercor.setups import (
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
from vercor.exchanges import (
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
    atm_grid = rectilinear("atm-grid", 128, 64, 0.0, 360.0, -90.0, 90.0)

    ocn_grid_shape = (64, 32)
    binary_mask = jax_ones(ocn_grid_shape).T.at[:2, :].set(0.0)  # land points
    ocn_grid = rectilinear(
        "ocn-grid", *ocn_grid_shape, 0.0, 360.0, -90.0, 90.0, mask=binary_mask
    )

    ice_grid = rectilinear("ice-grid", *ocn_grid_shape, 0.0, 360.0, -90.0, 90.0)
    lnd_grid = rectilinear("lnd-grid", 128, 64, 0.0, 360.0, -90.0, 90.0)

    # Build components
    atm = make_slab_atmosphere(atm_grid)
    ocn = make_slab_ocean(ocn_grid)
    ice = make_slab_seaice(ice_grid)
    lnd = make_slab_land(lnd_grid)

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=3600, steps=24)
    run_sequence = ["OCN", "ATM", "ICE", "LND"]

    # Coupler
    components: list[Any] = [atm, ocn, ice, lnd]
    cpl = Coupler.from_components(
        clock=clock,
        components=components,
        run_order=run_sequence,
    )

    # Exchanges
    # scalar fields (vector field))
    # ["SHF", "LHF", ("u10m", "v10m")]
    cpl.add_exchanges(
        (
            Exchange(
                source="ATM",
                target="OCN",
                fields=SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="OCN",
                target="ATM",
                fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="OCN",
                target="ICE",
                fields=OCEAN_TO_SEAICE_SURFACE_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="LND",
                target="ATM",
                fields=LAND_TO_ATMOSPHERE_SOIL_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
                regrid=conservative,
            ),
            Exchange(
                source="OCN",
                target="ICE",
                fields=OCEAN_TO_SEAICE_SURFACE_FIELDS,
                regrid=conservative,
            ),
            Exchange(
                source="ICE",
                target="OCN",
                fields=SEAICE_TO_OCEAN_FIELDS,
                regrid=conservative,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
    views = cpl.views(final_state, names=("ATM", "OCN", "LND", "ICE"))

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
