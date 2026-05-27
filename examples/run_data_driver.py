from datetime import datetime
from typing import Any, Callable

from vercor import Clock, RunSequence
from vercor.diagnostics import component_vector_speed
from vercor.setups.coupler_helpers import (
    ExchangeSpec,
    add_exchange_specs,
    build_coupler,
)
from vercor.setups.data.era5_atmosphere import make_era5_atmosphere
from vercor.setups.data.era5_land import make_era5_land
from vercor.setups.data.erainterim_ocean import make_erainterim_ocean
from vercor.setups.exchange_recipes import (
    ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
    ATMOSPHERE_TO_LAND_STATE_FIELDS,
    ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS,
    ATMOSPHERE_TO_OCEAN_STATE_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
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
    add_exchange_specs(
        cpl,
        (
            ExchangeSpec(
                source="ATM",
                destination="OCN",
                field_names=ATMOSPHERE_TO_OCEAN_STATE_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="ATM",
                destination="OCN",
                field_names=ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS,
                regridder_factory=conservative,
            ),
            ExchangeSpec(
                source="ATM",
                destination="LND",
                field_names=ATMOSPHERE_TO_LAND_STATE_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="ATM",
                destination="LND",
                field_names=ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
                regridder_factory=conservative,
            ),
            ExchangeSpec(
                source="OCN",
                destination="ATM",
                field_names=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="LND",
                destination="ATM",
                field_names=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
    views = cpl.runtime_component_views(final_state, names=("ATM", "OCN"))

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
        components=views,
        fields=variables,
        component_order=["ATM", "OCN"],
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            (
                "ATM",
                views["ATM"],
                total_surface_temperature,
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
        quiver_scale=150,
        cmap="coolwarm",
    )

    fig.colorbar(scalar_mappable, ax=axs, shrink=0.6)

    plt.show()
