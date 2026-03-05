from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from vercor import Clock, Coupler, Exchange
from vercor.components import JAXGCM, Land, Ocean
from vercor.coupler import RunSequence
from vercor.grid import RectilinearGrid
from vercor.regridders import bilinear, conservative

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

    # Inspect a few fields
    print("sst(OCN) mean:", np.nanmean(ocn.get("sea_surface_temperature")))
    print("sst(ATM) mean:", np.nanmean(atm.get("sea_surface_temperature")))
    print("temp mean:", np.nanmean(atm.get("temperature")))
    print("u_velocity(ATM) mean:", np.nanmean(atm.get("u_velocity")))
    print("v_velocity(ATM) mean:", np.nanmean(atm.get("v_velocity")))
    print("u_velocity(OCN) mean:", np.nanmean(ocn.get("u_velocity")))
    print("v_velocity(OCN) mean:", np.nanmean(ocn.get("v_velocity")))
    print("SOILM(LND) mean:", np.nanmean(lnd.get("soil_moisture")))
    print("SOILM(ATM) mean:", np.nanmean(atm.get("soil_moisture")))
    print("SHF(ATM) mean:", np.nanmean(atm.get("sensible_heat_flux")))
    print("SHF(LND) mean:", np.nanmean(lnd.get("sensible_heat_flux")))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

    lon_atm = np.array(atm.grid.longitude)
    lat_atm = np.array(atm.grid.latitude)
    longitude_source_2d, latitude_source_2d = np.meshgrid(
        lon_atm, lat_atm, indexing="ij"
    )
    scalar_source = atm.get("sea_surface_temperature").T
    u_source = atm.get("u_velocity").T
    v_source = atm.get("v_velocity").T

    lon_ocn = np.array(ocn.grid.longitude)
    lat_ocn = np.array(ocn.grid.latitude)
    longitude_target_2d, latitude_target_2d = np.meshgrid(
        lon_ocn, lat_ocn, indexing="ij"
    )
    scalar_target = ocn.get("sea_surface_temperature").T
    u_target = ocn.get("u_velocity").T
    v_target = ocn.get("v_velocity").T
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
