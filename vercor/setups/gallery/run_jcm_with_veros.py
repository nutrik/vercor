from datetime import datetime

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.dtypes import DTypePolicy
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import (
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerosConfig,
    make_veros_gcm,
)
from vercor.recipes import (
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


def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run this setup through the shared VerCOR CLI contract."""

    dtype = DTypePolicy(enable_x64=float_type == "float64")
    dtype._ensure_jax_capability()
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
            # JCM splits rain/snow using near-surface t0 before remapping;
            # total precipitation minus bulk evaporation supplies virtual salt flux.
            # Independently enable SST correction or 30-day SSS restoring.
            restore_to_climatology_temp=False,
            restore_to_climatology_salt=False,
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
                        "forc_salt_surface",
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
    # Gregorian continuation after two spinup days, ending 2100-01-01.
    clock = Clock(
        start=datetime(2000, 1, 3, 0, 0, 0),
        dt_seconds=86400.0,
        steps=(datetime(2100, 1, 1) - datetime(2000, 1, 3)).days,
        calendar="gregorian",
    )
    run_order = ["OCN", "LND", "ATM"]

    # Exchanges
    exchanges = (
        Exchange(
            quantity="intensive",
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
            regridder_factory=bilinear,
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
            fields=JCM_LAND_TO_ATMOSPHERE_FIELDS,
            regridder_factory=bilinear,
        ),
    )
    components = [ocn, lnd, atm]
    cpl = Coupler(
        clock=clock,
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        runtime=RuntimeOptions(dtype=dtype, topology=SurfaceMaskPolicy()),
        log_level=loglevel,
    )

    cpl.run(output=OutputTarget("."))


if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
