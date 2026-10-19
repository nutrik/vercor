from dataclasses import replace
from datetime import datetime

import matplotlib.pyplot as plt

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.dtypes import DTypePolicy
from vercor.components import Component, DataComponent
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


def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run this setup through the shared VerCOR CLI contract."""

    dtype = DTypePolicy(enable_x64=float_type == "float64")
    # Build components
    atm = make_era5_atmosphere()
    ocn = make_erainterim_ocean()
    lnd = make_era5_land()

    # These imports are displayed below; prescribed temperatures do not respond
    # to them. Declare the gallery's diagnostic role independently of factories.
    atm = DataComponent(
        atm.name,
        atm.grid,
        spec=replace(
            atm.spec, inputs=("sea_surface_temperature", "land_surface_temperature")
        ),
    )
    ocn = DataComponent(
        ocn.name,
        ocn.grid,
        spec=replace(
            ocn.spec,
            inputs=(
                "u_velocity",
                "v_velocity",
                "specific_humidity",
                "model_level_height",
                "density",
                "potential_temperature",
                "temperature",
                "net_shortwave_radiation_flux",
                "downward_longwave_radiation_flux",
            ),
        ),
    )
    lnd = DataComponent(
        lnd.name,
        lnd.grid,
        spec=replace(
            lnd.spec,
            inputs=(
                "specific_humidity",
                "model_level_height",
                "potential_temperature",
                "net_shortwave_radiation_flux",
                "downward_longwave_radiation_flux",
            ),
        ),
    )

    # Clock and sequence
    clock = Clock(start=datetime(2000, 1, 1, 0, 0, 0), dt_seconds=3600, steps=10)
    run_order = ["OCN", "ATM", "LND"]

    # Exchanges
    # scalar fields (vector field))
    # ["qbot", "zbot", ("ubot", "vbot")]
    exchanges = (
        Exchange(
            quantity="intensive",
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_OCEAN_STATE_FIELDS,
            route_id="atmosphere-ocean-state",
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS,
            route_id="atmosphere-ocean-radiation",
            regridder_factory=conservative,
        ),
        Exchange(
            quantity="intensive",
            source="ATM",
            target="LND",
            fields=ATMOSPHERE_TO_LAND_STATE_FIELDS,
            route_id="atmosphere-land-state",
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
            source="ATM",
            target="LND",
            fields=ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
            route_id="atmosphere-land-radiation",
            regridder_factory=conservative,
        ),
        Exchange(
            quantity="intensive",
            source="OCN",
            target="ATM",
            fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
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
        runtime=RuntimeOptions(dtype=dtype, topology=SurfaceMaskPolicy()),
        log_level=loglevel,
    )

    final_state = cpl.run(output=OutputTarget("."))
    views = final_state.components(("ATM", "OCN", "LND"))

    variables: list[tuple[ComponentMetric, str]] = [
        ("sea_surface_temperature", "sst"),
        ("specific_humidity", "qbot"),
        ("potential_temperature", "tbot"),
        ("model_level_height", "zbot"),
        ("temperature", "air_temperature"),
        ("density", "air_density"),
        ("net_shortwave_radiation_flux", "net_shortwave"),
        ("downward_longwave_radiation_flux", "downward_longwave"),
        (component_vector_speed, "speed"),
    ]

    print_component_field_means_table(
        components=views,
        fields=variables,
        component_order=["ATM", "OCN", "LND"],
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


if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
