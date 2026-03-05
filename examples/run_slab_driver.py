from datetime import datetime

import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import Atmosphere, Land, Ocean, SeaIce
from vercor.coupler import RunSequence
from vercor.regridders import (
    make_rectilinear_grid,
    bilinear,
    conservative,
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

    # Inspect a few fields
    print("sst(OCN) mean:", np.nanmean(ocn.get("sea_surface_temperature")))
    print("sst(ATM) mean:", np.nanmean(atm.get("sea_surface_temperature")))
    print("TA2M mean:", np.nanmean(atm.get("temperature_2m")))
    print("u10m mean:", np.nanmean(atm.get("u_velocity_10m")))
    print("v10m mean:", np.nanmean(atm.get("v_velocity_10m")))
    print("SOILM(LND) mean:", np.nanmean(lnd.get("soil_moisture")))
    print("SOILM(ATM) mean:", np.nanmean(atm.get("soil_moisture")))
    print("ICEFRAC mean:", np.nanmean(ice.get("ice_fraction")))
    print("SHF(ATM) mean:", np.nanmean(atm.get("sensible_heat_flux")))
    print("SHF(LND) mean:", np.nanmean(lnd.get("sensible_heat_flux")))

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

    lon_atm = np.array(atm.grid.longitude)
    lat_atm = np.array(atm.grid.latitude)
    longitude_source_2d, latitude_source_2d = np.meshgrid(
        lon_atm, lat_atm, indexing="ij"
    )
    scalar_source = atm.get("sea_surface_temperature").T
    u_source = atm.get("u_velocity_10m").T
    v_source = atm.get("v_velocity_10m").T

    lon_ocn = np.array(ocn.grid.longitude)
    lat_ocn = np.array(ocn.grid.latitude)
    longitude_target_2d, latitude_target_2d = np.meshgrid(
        lon_ocn, lat_ocn, indexing="ij"
    )
    scalar_target = ocn.get("sea_surface_temperature").T
    u_target = ocn.get("u_velocity_10m").T
    v_target = ocn.get("v_velocity_10m").T

    im = axs[0, 0].pcolormesh(
        longitude_source_2d,
        latitude_source_2d,
        scalar_source,
        shading="auto",
        cmap="coolwarm",
    )
    axs[0, 0].set_title("Initial Scalar Field")
    axs[0, 0].set_xlabel("Longitude")
    axs[0, 0].set_ylabel("Latitude")

    axs[0, 1].quiver(
        longitude_source_2d,
        latitude_source_2d,
        u_source,
        v_source,
        scale=100,
    )
    axs[0, 1].set_title("Initial Vector Field")
    axs[0, 1].set_xlabel("Longitude")
    axs[0, 1].set_ylabel("Latitude")

    axs[1, 0].pcolormesh(
        longitude_target_2d,
        latitude_target_2d,
        scalar_target,
        shading="auto",
        cmap="coolwarm",
    )
    axs[1, 0].set_title("Interpolated Scalar Field")
    axs[1, 0].set_xlabel("Longitude")
    axs[1, 0].set_ylabel("Latitude")

    axs[1, 1].quiver(
        longitude_target_2d,
        latitude_target_2d,
        u_target,
        v_target,
        scale=100,
    )
    axs[1, 1].set_title("Interpolated Vector Field")
    axs[1, 1].set_xlabel("Longitude")
    axs[1, 1].set_ylabel("Latitude")

    fig.colorbar(im, ax=axs, shrink=0.6)

    plt.show()
