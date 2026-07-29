from datetime import datetime

import matplotlib.pyplot as plt

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.components import Component
from vercor.diagnostics import (
    ComponentMetric,
    component_vector_speed,
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
    total_surface_temperature,
)
from vercor.output import OutputTarget
from vercor.regridding import bilinear, conservative
from vercor.topology import SurfaceMaskPolicy
from vercor.setups import make_era5_atmosphere
from vercor.setups import make_era5_land
from vercor.setups import make_erainterim_ocean
from vercor.recipes import (
    ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
    ATMOSPHERE_TO_LAND_STATE_FIELDS,
    ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS,
    ATMOSPHERE_TO_OCEAN_STATE_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)

if __name__ == "__main__":
    # Build components
    atm = make_era5_atmosphere()
    ocn = make_erainterim_ocean()
    lnd = make_era5_land()

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=3600, steps=10)
    run_order = ["OCN", "ATM", "LND"]

    # Exchanges
    # scalar fields (vector field))
    # ["qbot", "zbot", ("ubot", "vbot")]
    exchanges = (
        Exchange(
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_OCEAN_STATE_FIELDS,
            route_id="atmosphere-ocean-state",
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS,
            route_id="atmosphere-ocean-radiation",
            regridder_factory=conservative,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=ATMOSPHERE_TO_LAND_STATE_FIELDS,
            route_id="atmosphere-land-state",
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
            route_id="atmosphere-land-radiation",
            regridder_factory=conservative,
        ),
        Exchange(
            source="OCN",
            target="ATM",
            fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="LND",
            target="ATM",
            fields=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
    )
    components: list[Component] = [atm, ocn, lnd]
    cpl = Coupler(
        clock=clock,
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy()),
    )

    final_state = cpl.run(output=OutputTarget("."))
    views = final_state.components(("ATM", "OCN"))

    variables: list[tuple[ComponentMetric, str]] = [
        ("sea_surface_temperature", "sst"),
        ("specific_humidity", "qbot"),
        ("potential_temperature", "tbot"),
        ("model_level_height", "zbot"),
        (component_vector_speed, "speed"),
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
