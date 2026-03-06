from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import JAXGCM, Land, Ocean
from vercor.coupler import RunSequence
from vercor.grid import RectilinearGrid
from vercor.regridders import bilinear, conservative
from vercor.tools import (
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
)

from vercor.components.external.jax_gcm_tools import (
    generate_jcm_coords_forcing_topography_files,
)


if __name__ == "__main__":

    coords, terrain, forcing = generate_jcm_coords_forcing_topography_files()

    # Build components
    atm = JAXGCM(coords, terrain, forcing_data=forcing, jitted=True)

    ocn_binary_mask = np.where(terrain.fmask < 1, 1, 0).transpose()
    lnd_binary_mask = 1 - ocn_binary_mask

    hgrid = atm.model.coords.horizontal
    lnd_grid = RectilinearGrid(
        name="LND",
        longitude=np.rad2deg(hgrid.longitudes),
        latitude=np.rad2deg(hgrid.latitudes),
        binary_mask=lnd_binary_mask,
    )

    ocn_grid = RectilinearGrid(
        name="OCN",
        longitude=np.rad2deg(hgrid.longitudes),
        latitude=np.rad2deg(hgrid.latitudes),
        binary_mask=ocn_binary_mask,
    )

    ocn = Ocean(ocn_grid)
    lnd = Land(lnd_grid)

    if atm.grid.binary_mask is not None:
        print("Total number of grids = ", atm.grid.binary_mask.size)
        print("Sum of atm.grid.binary_mask = ", np.sum(atm.grid.binary_mask))

    if lnd.grid.binary_mask is not None:
        print("Sum of lnd.grid.binary_mask = ", np.sum(lnd.grid.binary_mask))

    if ocn.grid.binary_mask is not None:
        print("Sum of ocn.grid.binary_mask = ", np.sum(ocn.grid.binary_mask))

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=86400.0, steps=10)
    run_sequence = RunSequence(order=["OCN", "LND", "ATM"])

    # Coupler
    cpl = Coupler(clock=clock)
    components = [atm, ocn, lnd]
    for component in components:
        cpl.register(component)  # type: ignore

    cpl.set_components_run_sequence(run_sequence)

    # Exchanges
    # scalar fields (vector field))
    # ["SHF", "LHF", ("u10m", "v10m")]
    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=[
                ("u_velocity", "v_velocity"),
                "specific_humidity",
                "temperature",
                "net_shortwave_radiation_flux",
                "downward_longwave_radiation_flux",
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
            source="LND",
            destination="ATM",
            field_names=["soil_moisture", "land_surface_temperature"],
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

    cpl.initialize()
    cpl.run()
    cpl.finalize()

    # Inspect a few fields in a component-wise table.
    print_component_field_means_table(
        components={"ATM": atm, "OCN": ocn, "LND": lnd},
        fields=[
            ("sea_surface_temperature", "sst"),
            ("temperature", "temp"),
            ("u_velocity", "u_velocity"),
            ("v_velocity", "v_velocity"),
            ("soil_moisture", "soil_moisture"),
            ("sensible_heat_flux", "sensible_heat_flux"),
        ],
        component_order=["ATM", "OCN", "LND"],
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            ("ATM", atm, "sea_surface_temperature", "u_velocity", "v_velocity"),
            ("OCN", ocn, "sea_surface_temperature", "u_velocity", "v_velocity"),
        ],
        figsize=(15, 10),
        quiver_scale=100,
        cmap="coolwarm",
    )

    fig.colorbar(scalar_mappable, ax=axs, shrink=0.6)

    plt.show()
