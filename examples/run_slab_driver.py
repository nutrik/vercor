from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import Atmosphere, Land, Ocean, SeaIce
from vercor.coupler import RunSequence
from vercor.regridders import (
    make_rectilinear_grid,
    bilinear,
    conservative,
)
from vercor.tools import (
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
)

if __name__ == "__main__":
    # Build grids
    atm_grid = make_rectilinear_grid("atm-grid", 128, 64, 0.0, 360.0, -90.0, 90.0)

    ocn_grid_shape = (64, 32)
    binary_mask = np.ones(ocn_grid_shape).T
    binary_mask[:2, :] = 0.0  # land points
    ocn_grid = make_rectilinear_grid(
        "ocn-grid", *ocn_grid_shape, 0.0, 360.0, -90.0, 90.0, mask=binary_mask
    )

    ice_grid = make_rectilinear_grid(
        "ice-grid", *ocn_grid_shape, 0.0, 360.0, -90.0, 90.0
    )
    lnd_grid = make_rectilinear_grid("lnd-grid", 128, 64, 0.0, 360.0, -90.0, 90.0)

    # Build components
    atm = Atmosphere(atm_grid)
    ocn = Ocean(ocn_grid)
    ice = SeaIce(ice_grid)
    lnd = Land(lnd_grid)

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=3600, steps=24)
    run_sequence = RunSequence(order=["OCN", "ATM", "ICE", "LND"])

    # Coupler
    cpl = Coupler(clock=clock)
    components: list[Atmosphere | Ocean | SeaIce | Land] = [atm, ocn, ice, lnd]
    for component in components:
        cpl.register(component)

    cpl.set_components_run_sequence(run_sequence)

    # Exchanges
    # scalar fields (vector field))
    # ["SHF", "LHF", ("u10m", "v10m")]
    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=[
                ("u_velocity_10m", "v_velocity_10m"),
                "sensible_heat_flux",
                "latent_heat_flux",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["sea_surface_temperature"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="OCN",
            destination="ICE",
            field_names=["sea_surface_temperature"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="LND",
            destination="ATM",
            field_names=["soil_moisture"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=["latent_heat_flux", "sensible_heat_flux"],
            regridder_factory=conservative,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="OCN",
            destination="ICE",
            field_names=["sea_surface_temperature"],
            regridder_factory=conservative,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ICE",
            destination="OCN",
            field_names=["ice_fraction"],
            regridder_factory=conservative,
        )
    )

    cpl.initialize()
    cpl.run()
    cpl.finalize()

    # Inspect a few fields in a component-wise table.
    print_component_field_means_table(
        components={"ATM": atm, "OCN": ocn, "LND": lnd},
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
    print("ICE ice_fraction mean:", np.nanmean(ice.get("ice_fraction")))

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            (
                "ATM",
                atm,
                "sea_surface_temperature",
                "u_velocity_10m",
                "v_velocity_10m",
            ),
            (
                "OCN",
                ocn,
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
