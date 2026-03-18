from typing import Any, Callable
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from vercor import Clock, Coupler, Exchange
from vercor.components import ERA5Atmosphere, ERA5Land, ERAInterimOcean
from vercor.coupler import RunSequence
from vercor.regridders import bilinear, conservative
from vercor.tools import (
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Build components
    atm = ERA5Atmosphere()
    ocn = ERAInterimOcean()
    lnd = ERA5Land()

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=3600, steps=10)
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

    Metric = str | Callable[[Any], NDArray | float]

    variables: list[tuple[Metric, str]] = [
        ("sea_surface_temperature", "sst"),
        ("specific_humidity", "qbot"),
        ("potential_temperature", "tbot"),
        ("model_level_height", "zbot"),
        (
            lambda c: np.sqrt(c.get("u_velocity") ** 2 + c.get("v_velocity") ** 2),
            "speed",
        ),
    ]

    print_component_field_means_table(
        components={"ATM": atm, "OCN": ocn},
        fields=variables,
        component_order=["ATM", "OCN"],
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            ("ATM", atm, "total_surface_temperature", "u_velocity", "v_velocity"),
            ("OCN", ocn, "sea_surface_temperature", "u_velocity", "v_velocity"),
        ],
        figsize=(15, 10),
        quiver_scale=150,
        cmap="coolwarm",
    )

    fig.colorbar(scalar_mappable, ax=axs, shrink=0.6)

    plt.show()
