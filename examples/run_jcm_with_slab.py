from datetime import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt

from vercor import Clock, Coupler, Exchange
from vercor.setups import make_jax_gcm
from vercor.setups import make_slab_land
from vercor.setups import make_slab_ocean
from vercor.dtypes import as_jax_real_array
from vercor import RectilinearGrid
from vercor.regridding import bilinear, conservative
from vercor.exchanges import (
    ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.diagnostics import (
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
)

from vercor.setups.external.jax_gcm_tools import (
    generate_jcm_coords_forcing_topography_files,
)

if __name__ == "__main__":

    coords, terrain, forcing = generate_jcm_coords_forcing_topography_files()

    # Build components
    atm = make_jax_gcm(coords, terrain, forcing_data=forcing, jitted=True)

    ocn_binary_mask = jnp.where(as_jax_real_array(terrain.fmask) < 1, 1, 0).T
    lnd_binary_mask = 1 - ocn_binary_mask

    hgrid = coords.horizontal
    lnd_grid = RectilinearGrid(
        name="LND",
        longitude=jnp.rad2deg(as_jax_real_array(hgrid.longitudes)),
        latitude=jnp.rad2deg(as_jax_real_array(hgrid.latitudes)),
        binary_mask=lnd_binary_mask,
    )

    ocn_grid = RectilinearGrid(
        name="OCN",
        longitude=jnp.rad2deg(as_jax_real_array(hgrid.longitudes)),
        latitude=jnp.rad2deg(as_jax_real_array(hgrid.latitudes)),
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
    run_sequence = ["OCN", "LND", "ATM"]

    # Coupler
    components = [atm, ocn, lnd]
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
                fields=ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="OCN",
                target="ATM",
                fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="LND",
                target="ATM",
                fields=JCM_LAND_TO_ATMOSPHERE_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
                regrid=conservative,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
    views = cpl.views(final_state, names=("ATM", "OCN", "LND"))

    # Inspect a few fields in a component-wise table.
    print_component_field_means_table(
        components=views,
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
                views["ATM"],
                "sea_surface_temperature",
                "u_velocity",
                "v_velocity",
            ),
            (
                "OCN",
                views["OCN"],
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
