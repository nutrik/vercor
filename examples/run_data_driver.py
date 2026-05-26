from datetime import datetime
from typing import Any, Callable

from vercor import Clock, Exchange, RunSequence
from vercor.diagnostics import component_vector_speed
from vercor.setups.coupler_helpers import add_exchanges, build_coupler
from vercor.setups.data.era5_atmosphere import make_era5_atmosphere
from vercor.setups.data.era5_land import make_era5_land
from vercor.setups.data.erainterim_ocean import make_erainterim_ocean
from vercor.regridders import bilinear, conservative
from vercor.types import RuntimeArray
from vercor.diagnostics import (
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
    total_surface_temperature,
)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Build components
    atm = make_era5_atmosphere()
    ocn = make_erainterim_ocean()
    lnd = make_era5_land()

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=3600, steps=10)
    run_sequence = RunSequence(order=["OCN", "ATM", "LND"])

    # Coupler
    components: list[Any] = [atm, ocn, lnd]
    cpl = build_coupler(
        clock=clock,
        components=components,
        run_sequence=run_sequence,
    )

    # Exchanges
    # scalar fields (vector field))
    # ["qbot", "zbot", ("ubot", "vbot")]
    add_exchanges(
        cpl,
        (
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
            ),
            Exchange(
                source="ATM",
                destination="OCN",
                field_names=[
                    "net_shortwave_radiation_flux",
                    "downward_longwave_radiation_flux",
                ],
                regridder_factory=conservative,
            ),
            Exchange(
                source="ATM",
                destination="LND",
                field_names=[
                    "specific_humidity",
                    "model_level_height",
                    "potential_temperature",
                ],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="ATM",
                destination="LND",
                field_names=[
                    "net_shortwave_radiation_flux",
                    "downward_longwave_radiation_flux",
                ],
                regridder_factory=conservative,
            ),
            Exchange(
                source="OCN",
                destination="ATM",
                field_names=[
                    "sea_surface_temperature",
                ],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="LND",
                destination="ATM",
                field_names=[
                    "land_surface_temperature",
                ],
                regridder_factory=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)

    Metric = str | Callable[[Any], RuntimeArray | float]

    variables: list[tuple[Metric, str]] = [
        ("sea_surface_temperature", "sst"),
        ("specific_humidity", "qbot"),
        ("potential_temperature", "tbot"),
        ("model_level_height", "zbot"),
        (
            lambda component: component_vector_speed(component),
            "speed",
        ),
    ]

    print_component_field_means_table(
        components={
            "ATM": cpl.runtime_component_view(final_state, "ATM"),
            "OCN": cpl.runtime_component_view(final_state, "OCN"),
        },
        fields=variables,
        component_order=["ATM", "OCN"],
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            (
                "ATM",
                cpl.runtime_component_view(final_state, "ATM"),
                total_surface_temperature,
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
        quiver_scale=150,
        cmap="coolwarm",
    )

    fig.colorbar(scalar_mappable, ax=axs, shrink=0.6)

    plt.show()
