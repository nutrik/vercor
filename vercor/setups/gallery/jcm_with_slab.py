from datetime import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RectilinearGrid,
    RuntimeOptions,
)
from vercor.setups import (
    JCMAtmosphereConfig,
    load_jcm_inputs,
    make_jcm_atmosphere,
    make_slab_land,
    make_slab_ocean,
)
from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.output import OutputTarget
from vercor.regridding import bilinear, conservative
from vercor.recipes import (
    ATMOSPHERE_TO_LAND_LATENT_HEAT_FLUX_FIELDS,
    ATMOSPHERE_TO_OCEAN_TURBULENT_HEAT_FLUX_FIELDS,
    LAND_TO_ATMOSPHERE_STATE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.diagnostics import (
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
)


def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run this setup through the shared VerCOR CLI contract."""

    dtype = DTypePolicy(enable_x64=float_type == "float64")
    dtype._ensure_jax_capability()

    inputs = load_jcm_inputs()
    coords = inputs.coords
    terrain = inputs.terrain
    forcing = inputs.forcing

    # Build components
    atm = make_jcm_atmosphere(
        coords,
        terrain,
        config=JCMAtmosphereConfig(forcing_data=forcing, jitted=True),
    )

    ocean_binary_mask = jnp.where(as_jax_real_array(terrain.fmask) < 1, 1, 0).T
    land_fraction = as_jax_real_array(terrain.fmask).T
    lnd_binary_mask = jnp.where(land_fraction > 0, 1, 0)

    hgrid = coords.horizontal
    lnd_grid = RectilinearGrid.from_coordinates(
        "LND",
        longitude=jnp.rad2deg(as_jax_real_array(hgrid.longitudes)),
        latitude=jnp.rad2deg(as_jax_real_array(hgrid.latitudes)),
        binary_mask=lnd_binary_mask,
    )

    ocn_grid = RectilinearGrid.from_coordinates(
        "OCN",
        longitude=jnp.rad2deg(as_jax_real_array(hgrid.longitudes)),
        latitude=jnp.rad2deg(as_jax_real_array(hgrid.latitudes)),
        binary_mask=ocean_binary_mask,
    )

    ocn = make_slab_ocean(ocn_grid, fractional_coverage=1 - land_fraction)
    lnd = make_slab_land(lnd_grid, fractional_coverage=land_fraction)

    if atm.grid.binary_mask is not None:
        print("Total number of grids = ", atm.grid.binary_mask.size)
        print("Sum of atm.grid.binary_mask = ", float(jnp.sum(atm.grid.binary_mask)))

    if lnd.grid.binary_mask is not None:
        print("Sum of lnd.grid.binary_mask = ", float(jnp.sum(lnd.grid.binary_mask)))

    if ocn.grid.binary_mask is not None:
        print("Sum of ocn.grid.binary_mask = ", float(jnp.sum(ocn.grid.binary_mask)))

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=86400.0, steps=10)
    run_order = [ocn.name, lnd.name, atm.name]

    # JCM 3 supplies grid-mean turbulent fluxes. This toy slab example uses
    # those densities on both tiles; it does not resolve tile-specific budgets.
    # Exchanges
    # scalar fields (vector field))
    # ["SHF", "LHF", ("u10m", "v10m")]
    exchanges = (
        Exchange(
            quantity="intensive",
            source=atm.name,
            target=ocn.name,
            fields=ATMOSPHERE_TO_OCEAN_TURBULENT_HEAT_FLUX_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
            source=ocn.name,
            target=atm.name,
            fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
            source=lnd.name,
            target=atm.name,
            fields=LAND_TO_ATMOSPHERE_STATE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
            source=atm.name,
            target=lnd.name,
            fields=ATMOSPHERE_TO_LAND_LATENT_HEAT_FLUX_FIELDS,
            regridder_factory=conservative,
        ),
    )
    components = [atm, ocn, lnd]
    cpl = Coupler(
        clock=clock,
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        # Shared native grid: each component owns the native tile fraction.
        # The ocean-binary-derived SurfaceMaskPolicy is for disjoint wet grids.
        runtime=RuntimeOptions(dtype=dtype, topology=None),
        log_level=loglevel,
    )

    final_state = cpl.run(output=OutputTarget("."))
    views = final_state.components((atm.name, ocn.name, lnd.name))

    # Inspect a few fields in a component-wise table.
    print_component_field_means_table(
        components=views,
        fields=[
            ("sea_surface_temperature", "sst"),
            ("air_temperature", "temp"),
            ("air_u_velocity", "air_u_velocity"),
            ("air_v_velocity", "air_v_velocity"),
            ("soil_moisture", "soil_moisture"),
            ("sensible_heat_flux", "sensible_heat_flux"),
        ],
        component_order=[atm.name, ocn.name, lnd.name],
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            (
                atm.name,
                views[atm.name],
                "sea_surface_temperature",
                "air_u_velocity",
                "air_v_velocity",
            ),
        ],
        figsize=(15, 10),
        quiver_scale=100,
        cmap="coolwarm",
    )

    fig.colorbar(scalar_mappable, ax=axs, shrink=0.6)

    plt.show()


if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
