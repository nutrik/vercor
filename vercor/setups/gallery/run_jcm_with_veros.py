from datetime import datetime

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import (
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerosConfig,
    make_veros_gcm,
)
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.setups import make_jcm_land_atmosphere
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

from jcm.physics.speedy.params import Parameters


def _default_jcm_parameter_values(
    parameters: list[str],
    default_parameters: Parameters,
) -> dict[str, float]:
    """Return selected default JCM parameter values for the example script."""

    output = {}
    for parameter in parameters:
        parameter_group_name, parameter_name = parameter.split(".")
        output[parameter] = getattr(
            getattr(default_parameters, parameter_group_name),
            parameter_name,
        )
    return output


if __name__ == "__main__":
    optimized_parameters: list = [
        "surface_flux.vgust",
        "convection.rhbl",
        "condensation.rhlsc",
        "surface_flux.cds",
    ]

    custom_jcm_parameters: dict[str, float] = _default_jcm_parameter_values(
        parameters=optimized_parameters,
        default_parameters=Parameters.default(),
    )

    # change the values of the parameters to be optimized here
    # custom_jcm_parameters['surface_flux.vgust'] = 5.01

    ocn = make_veros_gcm(
        config=VerosConfig(
            spinup=Spinup(enabled=True),
            output=OutputSpec(
                period=PeriodOutput(
                    frequency="month",
                    variables=(
                        "temp",
                        "salt",
                        "u",
                        "v",
                        "w",
                        "surface_taux",
                        "surface_tauy",
                        "psi",
                    ),
                ),
            ),
        ),
    )

    jcm_setup = make_jcm_land_atmosphere(
        ocn.grid,
        config=JCMLandAtmosphereConfig(
            atmosphere=JAXGCMConfig(
                custom_parameters=custom_jcm_parameters,
                spinup=Spinup(enabled=True),
                output=OutputSpec(period=PeriodOutput(frequency="month")),
                jitted=True,
            ),
        ),
    )
    lnd = jcm_setup.land
    atm = jcm_setup.atmosphere

    # Clock and sequence
    # Note that the number of steps is set to 365*100-2,
    # which corresponds to 100 years of simulation with a daily time step,
    # starting from January 3rd, 2000.
    # The -2 accounts for the fact that the simulation starts on January 3rd,
    # because of 2 days spinup of JCM & Veros models,
    # so it will end on December 31st, 2099.
    clock = Clock(
        start=datetime(2000, 1, 3, 0, 0, 0),
        dt_seconds=86400.0,
        steps=365 * 100 - 2,
        calendar="noleap",
    )
    run_order = ["OCN", "LND", "ATM"]

    # Exchanges
    exchanges = (
        Exchange(
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
            regridder_factory=bilinear,
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
            fields=JCM_LAND_TO_ATMOSPHERE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
            regridder_factory=bilinear,
        ),
    )
    components = [ocn, lnd, atm]
    cpl = Coupler(
        clock=clock,
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy()),
    )

    cpl.run(output=OutputTarget("."))
