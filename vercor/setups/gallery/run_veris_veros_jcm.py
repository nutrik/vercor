"""Couple JCM land/atmosphere, native Veris sea ice and prognostic Veros.

Each interval runs LND -> CICE -> OCN -> ATM. Ice uses the preceding air
snapshot, exports total ocean boundary rates once, and supplies current per-ice
surface properties. The ocean supplies (1-A)*current SST after its advance.
Conservative surface contributions are composed on JCM's grid before native
physics applies the separate land/marine fractions. JCM has one effective
marine temperature; this does not resolve separate ice/water turbulent fluxes.
"""

from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING
from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.recipes import (
    JCM_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
    JCM_ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_JCM_OPEN_WATER_FIELDS,
    OCEAN_TO_VERIS_FORCING_FIELDS,
    VERIS_TO_JCM_COVERAGE_FIELDS,
    VERIS_TO_JCM_SURFACE_FIELDS,
    VERIS_TO_OCEAN_FLUX_FIELDS,
)
from vercor.regridding import bilinear, conservative
from vercor.setups import (
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerisConfig,
    VerosConfig,
    make_jcm_land_atmosphere,
    make_veris_veros,
)
from vercor.topology import SurfaceMaskPolicy

if TYPE_CHECKING:
    from vercor.setups import JCMInputs, VerisVerosSetup


def build_coupler(
    *,
    pair: VerisVerosSetup | None = None,
    jcm_inputs: JCMInputs | None = None,
    clock: Clock | None = None,
    veris_config: VerisConfig | None = None,
    ocean_config: VerosConfig | None = None,
    jcm_config: JCMLandAtmosphereConfig | None = None,
    log_level: int | str = "INFO",
    dtype: DTypePolicy | None = None,
) -> Coupler:
    """Build the four components once, with injectable native inputs and clock.

    Explicit configurations must enable model-level ice forcing, total ocean
    feedback/current water contribution and coupled JCM surfaces. An injected
    pair must expose those contracts on the same grid and cannot also receive
    pair configuration overrides. The native SPEEDY calendar requires noleap.
    """
    clock = (
        Clock(datetime(2000, 1, 3), 86400.0, 365, calendar="noleap")
        if clock is None
        else clock
    )
    if clock.calendar != "noleap":
        raise ValueError("Coupled JCM requires a noleap clock")
    if pair is not None and (veris_config is not None or ocean_config is not None):
        raise ValueError(
            "Injected pair cannot also receive pair configuration overrides"
        )
    if veris_config is not None and veris_config.atmosphere_forcing != "model_level":
        raise ValueError("Veris requires model_level atmosphere forcing")
    if ocean_config is not None and (
        not ocean_config.export_open_water_surface
        or ocean_config.surface_forcing != "coupled"
        or ocean_config.setup != "global_4deg"
    ):
        raise ValueError(
            "Veros requires global_4deg coupled forcing and export_open_water_surface"
        )
    if jcm_config is not None and not jcm_config.atmosphere.seaice_coupling:
        raise ValueError("JCM requires seaice_coupling")
    dtype = DTypePolicy(enable_x64=True) if dtype is None else dtype
    dtype._ensure_jax_capability()
    if pair is None:
        pair = make_veris_veros(
            config=veris_config
            or VerisConfig(
                atmosphere_forcing="model_level",
                output=OutputSpec(period=PeriodOutput(frequency="month")),
            ),
            ocean_config=ocean_config
            or VerosConfig(
                surface_forcing="coupled",
                export_open_water_surface=True,
                output=OutputSpec(
                    period=PeriodOutput(
                        frequency="month", variables=("temp", "salt", "u", "v", "psi")
                    )
                ),
            ),
        )
    ice, ocean = pair.seaice, pair.ocean
    if ice.grid is not ocean.grid:
        raise ValueError("Injected ice/ocean pair must share the exact native grid")
    required_ice = {
        "open_water_albedo",
        "surface_pressure",
        "downward_shortwave_radiation_flux",
    }
    if (
        not required_ice.issubset(ice.spec.inputs)
        or "u_velocity_10m" in ice.spec.inputs
    ):
        raise ValueError("Injected Veris requires model_level atmosphere forcing")
    if (
        "open_water_temperature_contribution" not in ocean.spec.outputs
        or "ice_fraction" not in ocean.spec.inputs
    ):
        raise ValueError("Injected ocean requires export_open_water_surface")
    if not {"ice_surface_temperature", "ice_surface_albedo", "ice_fraction"}.issubset(
        ice.spec.outputs
    ):
        raise ValueError("Injected ice must export current surface properties")
    jcm = make_jcm_land_atmosphere(
        ocean.grid,
        inputs=jcm_inputs,
        config=jcm_config
        or JCMLandAtmosphereConfig(
            atmosphere=JAXGCMConfig(
                seaice_coupling=True,
                spinup=Spinup(enabled=True),
                output=OutputSpec(period=PeriodOutput(frequency="month")),
            )
        ),
    )
    land, atmosphere = jcm.land, jcm.atmosphere
    return Coupler(
        clock,
        components=(land, ice, ocean, atmosphere),
        run_order=(land.name, ice.name, ocean.name, atmosphere.name),
        exchanges=(
            Exchange(
                atmosphere.name,
                ice.name,
                JCM_ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
                quantity="intensive",
                regridder_factory=bilinear,
            ),
            Exchange(
                ocean.name,
                ice.name,
                OCEAN_TO_VERIS_FORCING_FIELDS,
                quantity="intensive",
                source_coverage="binary",
                regridder_factory=bilinear,
            ),
            Exchange(
                ice.name,
                ocean.name,
                VERIS_TO_OCEAN_FLUX_FIELDS[:-1],
                route_id=f"{ice.name}->{ocean.name}.scalars",
                quantity="source_flux",
                source_coverage="binary",
                regridder_factory=conservative,
            ),
            Exchange(
                ice.name,
                ocean.name,
                VERIS_TO_OCEAN_FLUX_FIELDS[-1:],
                route_id=f"{ice.name}->{ocean.name}.stress",
                quantity="source_flux",
                source_coverage="binary",
                regridder_factory=bilinear,
            ),
            Exchange(
                ice.name,
                ocean.name,
                VERIS_TO_JCM_COVERAGE_FIELDS,
                route_id=f"{ice.name}->{ocean.name}.coverage",
                quantity="intensive",
                source_coverage="binary",
                regridder_factory=bilinear,
            ),
            Exchange(
                ocean.name,
                atmosphere.name,
                OCEAN_TO_JCM_OPEN_WATER_FIELDS,
                quantity="source_flux",
                source_coverage="binary",
                regridder_factory=conservative,
            ),
            Exchange(
                ice.name,
                atmosphere.name,
                VERIS_TO_JCM_SURFACE_FIELDS,
                route_id=f"{ice.name}->{atmosphere.name}.surface",
                quantity="source_flux",
                regridder_factory=conservative,
            ),
            Exchange(
                ice.name,
                atmosphere.name,
                VERIS_TO_JCM_COVERAGE_FIELDS,
                route_id=f"{ice.name}->{atmosphere.name}.coverage",
                quantity="source_flux",
                source_coverage="binary",
                regridder_factory=conservative,
            ),
            Exchange(
                land.name,
                atmosphere.name,
                JCM_LAND_TO_ATMOSPHERE_FIELDS,
                quantity="intensive",
                regridder_factory=bilinear,
            ),
            Exchange(
                atmosphere.name,
                land.name,
                JCM_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
                quantity="intensive",
                regridder_factory=bilinear,
            ),
        ),
        runtime=RuntimeOptions(
            dtype=dtype,
            topology=SurfaceMaskPolicy(
                atmosphere=atmosphere.name, ocean=ocean.name, land=land.name
            ),
        ),
        log_level=log_level,
    )


def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run the packaged gallery through the shared VerCOR CLI."""
    if float_type not in {"float32", "float64"}:
        raise ValueError("float_type must be float32 or float64")
    build_coupler(
        log_level=loglevel, dtype=DTypePolicy(enable_x64=float_type == "float64")
    ).run(output=OutputTarget("."))


if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
