from datetime import datetime
from typing import Any, Callable

import jax

from setups.jax_array_helpers import component_vector_speed
from vercor import Clock, Coupler, Exchange, RunSequence
from setups.data.era5_atmosphere import make_era5_atmosphere
from setups.data.era5_land import make_era5_land
from setups.data.erainterim_ocean import make_erainterim_ocean
from vercor.regridders import bilinear, conservative
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
    cpl = Coupler(clock=clock)
    components: list[Any] = [atm, ocn, lnd]
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
    final_state = cpl.run()
    cpl.finalize(final_state)

    Metric = str | Callable[[Any], jax.Array | float]

    variables: list[tuple[Metric, str]] = [
        ("sea_surface_temperature", "sst"),
        ("specific_humidity", "qbot"),
        ("potential_temperature", "tbot"),
        ("model_level_height", "zbot"),
        (
            component_vector_speed,
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
