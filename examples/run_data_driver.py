from datetime import datetime

import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import ERA5Atmosphere, ERA5Land, ERAInterimOcean
from vercor.coupler import RunSequence
from vercor.regridders import bilinear, conservative


if __name__ == "__main__":
    # Build components
    atm = ERA5Atmosphere()
    ocn = ERAInterimOcean()
    lnd = ERA5Land()

    # Clock and sequence
    clock = Clock(start=datetime(2025, 1, 1, 0, 0, 0), dt_seconds=3600, steps=10)
    run_sequence = RunSequence(order=["OCN", "ATM", "LND"])

    # Coupler
    cpl = Coupler(clock=clock)
    components: list[ERA5Atmosphere | ERAInterimOcean | ERA5Land] = [atm, ocn, lnd]
    for component in components:
        cpl.register(component)

    cpl.set_components_run_sequence(run_sequence)

    # Exchanges
    # scalar fields (vector field))
    # ["qbot", "zbot", ("ubot", "vbot")]
    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=[
                ("u_velocity", "v_velocity"),
                "specific_humidity",
                "model_level_height",
                "density",
                "potential_temperature",
                "temperature",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=[
                "net_shortwave_radiation_flux",
                "downward_longwave_radiation_flux",
            ],
            regridder_factory=conservative,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=[
                "specific_humidity",
                "model_level_height",
                "potential_temperature",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=[
                "net_shortwave_radiation_flux",
                "downward_longwave_radiation_flux",
            ],
            regridder_factory=conservative,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=[
                "sea_surface_temperature",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="LND",
            destination="ATM",
            field_names=[
                "land_surface_temperature",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.initialize()
    cpl.run()
    cpl.finalize()

    print("sst(OCN) mean:", np.nanmin(ocn.get("sea_surface_temperature")))
    print("sst(ERA) mean:", np.nanmin(atm.get("sea_surface_temperature")))
    print("qbot(ERA) mean:", np.nanmin(atm.get("specific_humidity")))
    print("qbot(OCN) mean:", np.nanmin(ocn.get("specific_humidity")))
    print("tbot(ERA) mean:", np.nanmin(atm.get("potential_temperature")))
    print("tbot(OCN) mean:", np.nanmin(ocn.get("potential_temperature")))
    print("zbot(ERA) mean:", np.nanmin(atm.get("model_level_height")))
    print("zbot(OCN) mean:", np.nanmin(ocn.get("model_level_height")))
    print(
        "speed(ERA) mean:",
        np.nanmean(np.sqrt(atm.get("u_velocity") ** 2 + atm.get("v_velocity") ** 2)),
    )
    print(
        "speed(OCN) mean:",
        np.nanmean(np.sqrt(ocn.get("u_velocity") ** 2 + ocn.get("v_velocity") ** 2)),
    )
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

    lon_atm = np.array(atm.grid.longitude)
    lat_atm = np.array(atm.grid.latitude)
    longitude_source_2d, latitude_source_2d = np.meshgrid(
        lon_atm, lat_atm, indexing="ij"
    )
    scalar_source = atm.get("total_surface_temperature").T
    # scalar_source = atm.get("net_shortwave_radiation_flux").T
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
        vmin=220,
        vmax=310,
    )
    axs[0, 0].set_title("Initial Scalar Field")
    axs[0, 0].set_xlabel("Longitude")
    axs[0, 0].set_ylabel("Latitude")

    axs[0, 1].quiver(
        longitude_source_2d,
        latitude_source_2d,
        u_source,
        v_source,
        scale=150,
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
        vmin=220,
        vmax=310,
    )
    axs[1, 0].set_title("Interpolated Scalar Field")
    axs[1, 0].set_xlabel("Longitude")
    axs[1, 0].set_ylabel("Latitude")

    axs[1, 1].quiver(
        longitude_target_2d,
        latitude_target_2d,
        u_target,
        v_target,
        scale=150,
    )
    axs[1, 1].set_title("Interpolated Vector Field")
    axs[1, 1].set_xlabel("Longitude")
    axs[1, 1].set_ylabel("Latitude")

    fig.colorbar(im, ax=axs, shrink=0.6)

    plt.show()
