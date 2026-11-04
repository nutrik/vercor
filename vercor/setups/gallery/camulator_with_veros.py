from datetime import datetime, timedelta
import os

import xarray as xr
import yaml

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.dtypes import DTypePolicy
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import CAMulatorConfig, Spinup, VerosConfig, make_camulator_gcm
from vercor.setups import make_camulator_land
from vercor.setups import make_veros_gcm
from vercor.recipes import (
    ATMOSPHERE_TO_OCEAN_BULK_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)

from vercor.regridding import bilinear, conservative
from vercor.topology import SurfaceMaskPolicy


def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run this setup through the shared VerCOR CLI contract."""

    # Select a vapor-trained native configuration/checkpoint explicitly. The
    # published Qtot-only model cannot supply physical water-vapor humidity.
    config_path = "/glade/u/home/rnuterman/veros_coupling/climate/camulator_config.yml"
    water_vapor_variable = os.environ.get("VERCOR_CAMULATOR_WATER_VAPOR_VARIABLE")
    if not water_vapor_variable:
        raise ValueError(
            "CAMulator gallery requires VERCOR_CAMULATOR_WATER_VAPOR_VARIABLE "
            "naming a real native vapor channel."
        )
    with open(config_path) as stream:
        native_conf = yaml.safe_load(stream)
    if (
        water_vapor_variable.casefold() == "qtot"
        or water_vapor_variable not in native_conf["data"]["variables"]
    ):
        raise ValueError(
            "CAMulator gallery requires a vapor-specific prognostic channel in both "
            "native input and prediction; Qtot is total water."
        )
    initial_time = datetime.fromisoformat(str(native_conf["predict"]["start_datetime"]))
    diagnostics_path = os.environ.get("VERCOR_CAMULATOR_INITIAL_DIAGNOSTICS")
    if not diagnostics_path:
        raise ValueError(
            "CAMulator gallery requires VERCOR_CAMULATOR_INITIAL_DIAGNOSTICS pointing "
            "to timestamped native TS/FSNS/FLNS initial-condition data."
        )
    with xr.open_dataset(diagnostics_path) as initial_data:
        # Exact selection: never nearest-time interpolation or fabricated weather.
        selected = initial_data.sel(time=initial_time)
        initial_diagnostics: dict[str, object] = {"time": initial_time}
        for name in ("TS", "FSNS", "FLNS"):
            initial_diagnostics[name] = (
                selected[name].transpose("latitude", "longitude").values.copy()
            )

    dtype = DTypePolicy(enable_x64=float_type == "float64")
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
            custom_parameters={"dt_tracer": timedelta(hours=6).total_seconds()},
        ),
    )

    atm = make_camulator_gcm(
        config=CAMulatorConfig(
            config_path=config_path,
            initial_diagnostics=initial_diagnostics,
            water_vapor_variable=water_vapor_variable,
            model_weights_path="/glade/u/home/rnuterman/veros_coupling/climate/checkpoint.pt00091.pt",
            output=OutputSpec(period=PeriodOutput(frequency="month")),
        ),
    )

    lnd = make_camulator_land(
        config_path="/glade/u/home/rnuterman/veros_coupling/climate/camulator_land_config.yml",
        camulator_grid=atm.grid,
        ocn_grid=ocn.grid,
    )

    clock = Clock(
        start=initial_time,
        dt_seconds=86400.0 // 4,
        steps=365 - 2 * 4,
        calendar="gregorian",
    )
    run_order = [ocn.name, lnd.name, atm.name]

    # Exchanges
    exchanges = (
        Exchange(
            quantity="intensive",
            source=atm.name,
            target=ocn.name,
            fields=ATMOSPHERE_TO_OCEAN_BULK_FORCING_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="source_flux",
            source=ocn.name,
            target=atm.name,
            fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=conservative,
        ),
        Exchange(
            quantity="source_flux",
            source=lnd.name,
            target=atm.name,
            fields=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=conservative,
        ),
    )
    components = [ocn, lnd, atm]
    cpl = Coupler(
        clock=clock,
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        runtime=RuntimeOptions(
            dtype=dtype,
            topology=SurfaceMaskPolicy(
                atmosphere=atm.name, ocean=ocn.name, land=lnd.name
            ),
        ),
        log_level=loglevel,
    )

    cpl.run(output=OutputTarget("."))


if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
