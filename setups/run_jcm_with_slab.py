from datetime import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt

from vercor import Clock, Exchange, RunSequence
from setups.coupler_helpers import add_exchanges, build_coupler
from setups.external.jax_gcm import make_jax_gcm
from setups.slab.land import make_slab_land
from setups.slab.ocean import make_slab_ocean
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid
from vercor.regridders import bilinear, conservative
from vercor.diagnostics import (
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
)

from setups.external.jax_gcm_tools import (
    generate_jcm_coords_forcing_topography_files,
)

if __name__ == "__main__":

    coords, terrain, forcing = generate_jcm_coords_forcing_topography_files()

    # Build components
    atm = make_jax_gcm(coords, terrain, forcing_data=forcing, jitted=True)

    ocn_binary_mask = jnp.where(as_jax_real_array(terrain.fmask) < 1, 1, 0).T
    lnd_binary_mask = 1 - ocn_binary_mask

    lnd_grid = RectilinearGrid(
        name="LND",
        longitude=atm.grid.longitude,
        latitude=atm.grid.latitude,
        binary_mask=lnd_binary_mask,
    )

    ocn_grid = RectilinearGrid(
        name="OCN",
        longitude=atm.grid.longitude,
        latitude=atm.grid.latitude,
        binary_mask=ocn_binary_mask,
    )

    ocn = make_slab_ocean(ocn_grid)
    lnd = make_slab_land(lnd_grid)

    if atm.grid.binary_mask is not None:
        print("Total number of grids = ", atm.grid.binary_mask.size)
        print("Sum of atm.grid.binary_mask = ", float(jnp.sum(atm.grid.binary_mask)))

    if lnd.grid.binary_mask is not None:
        print("Sum of lnd.grid.binary_mask = ", float(jnp.sum(lnd.grid.binary_mask)))

    if ocn.grid.binary_mask is not None:
        print("Sum of ocn.grid.binary_mask = ", float(jnp.sum(ocn.grid.binary_mask)))

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=86400.0, steps=10)
    run_sequence = RunSequence(order=["OCN", "LND", "ATM"])

    # Coupler
    components = [atm, ocn, lnd]
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
                    ("u_velocity", "v_velocity"),
                    "specific_humidity",
                    "temperature",
                    "net_shortwave_radiation_flux",
                    "downward_longwave_radiation_flux",
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
                source="LND",
                destination="ATM",
                field_names=["soil_moisture", "land_surface_temperature"],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="ATM",
                destination="LND",
                field_names=["latent_heat_flux", "sensible_heat_flux"],
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
            (
                "ATM",
                cpl.runtime_component_view(final_state, "ATM"),
                "sea_surface_temperature",
                "u_velocity",
                "v_velocity",
            ),
            (
                "OCN",
                cpl.runtime_component_view(final_state, "OCN"),
                "sea_surface_temperature",
                "u_velocity",
                "v_velocity",
            ),
        ],
        figsize=(15, 10),
        quiver_scale=100,
        cmap="coolwarm",
    )

    fig.colorbar(scalar_mappable, ax=axs, shrink=0.6)

    plt.show()
