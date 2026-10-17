"""Check the packaged assembly and real global Veris/Veros-data/ERA5 runs."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from click.testing import CliRunner
import pytest
import yaml

from tests.test_shared_surface_fluxes import surface_fields
from tests.test_veris_component import small_ocean
from vercor import Coupler
from vercor.cli import cli
from vercor.components import (
    CallableComponent,
    ComponentMaskSpec,
    ComponentSpec,
    DataComponent,
)
from vercor.fields import _flatten_field_items
from vercor.output import OutputTarget
from vercor.recipes import (
    ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
    OCEAN_TO_VERIS_FORCING_FIELDS,
)


@pytest.mark.fast_always
@pytest.mark.parametrize("float_type", ["float32", "float64"])
def test_gallery_assembles_exact_sequence_and_prescribed_forcing(
    monkeypatch: pytest.MonkeyPatch, float_type: str
) -> None:
    gallery = importlib.import_module("vercor.setups.gallery.run_veris_dataveros_era5")
    grid, _ = small_ocean()
    all_fields = surface_fields()
    ocean = DataComponent(
        "OCN",
        grid,
        {
            name: all_fields[name]
            for name in _flatten_field_items(OCEAN_TO_VERIS_FORCING_FIELDS)
        },
    )
    atmosphere = DataComponent(
        "ATM",
        grid,
        {
            name: all_fields[name]
            for name in _flatten_field_items(ATMOSPHERE_TO_VERIS_FORCING_FIELDS)
        },
    )
    land = DataComponent("LND", grid, {"land_surface_temperature": 273.15})
    ice = CallableComponent(
        "CICE",
        grid,
        lambda f: {},
        spec=ComponentSpec(
            inputs=tuple(all_fields),
            outputs=("ice_fraction",),
            initial_fields={"ice_fraction": 0.0},
            masks=ComponentMaskSpec(fractional_field="ice_fraction"),
        ),
    )
    captured: dict[str, Any] = {}

    def paired(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return type("Pair", (), {"seaice": ice, "ocean": ocean})()

    def coupler(*args: Any, **kwargs: Any) -> Coupler:
        captured["coupler"] = kwargs
        return Coupler(*args, **kwargs)

    monkeypatch.setattr(gallery, "make_veris_verosdata", paired)
    monkeypatch.setattr(gallery, "make_era5_atmosphere", lambda **kwargs: atmosphere)
    monkeypatch.setattr(gallery, "make_era5_land", lambda **kwargs: land)
    monkeypatch.setattr(gallery, "Coupler", coupler)
    result = gallery._make_coupler(loglevel="error", float_type=float_type, steps=2)
    assert tuple(result.run_order) == ("LND", "CICE", "OCN", "ATM")
    assert result.clock.steps == 2 and result.clock.dt_seconds == 86400.0
    assert result.clock.calendar == "noleap"
    assert ice.grid is ocean.grid
    assert captured["coupler"]["log_level"] == "error"
    assert captured["coupler"]["runtime"].dtype.enable_x64 == (float_type == "float64")
    assert captured["coupler"]["runtime"].topology is None
    assert captured["config"].model_timestep.total_seconds() == 3600
    assert captured["config"].output.period.frequency == "month"
    assert tuple(
        (route.source, route.target, route.quantity) for route in result.exchanges
    ) == (("ATM", "CICE", "intensive"), ("OCN", "CICE", "intensive"))
    assert result.exchanges[0].fields == ATMOSPHERE_TO_VERIS_FORCING_FIELDS
    assert result.exchanges[1].fields == OCEAN_TO_VERIS_FORCING_FIELDS


@pytest.mark.fast_always
def test_gallery_run_contract_and_cli_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gallery = importlib.import_module("vercor.setups.gallery.run_veris_dataveros_era5")
    captured: dict[str, Any] = {}

    @dataclass
    class Runner:
        def run(self, *, output: OutputTarget) -> None:
            captured["output"] = output

    def assemble(**kwargs: Any) -> Runner:
        captured.update(kwargs)
        return Runner()

    monkeypatch.setattr(gallery, "_make_coupler", assemble)
    gallery.run_setup(loglevel="warning", float_type="float32")
    assert captured["loglevel"] == "warning" and captured["float_type"] == "float32"
    assert captured["output"].directory == Path(".")
    result = CliRunner().invoke(
        cli, ["copy-setup", "run_veris_dataveros_era5", "--to", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    copied = tmp_path / "run_veris_dataveros_era5.py"
    assert isinstance(gallery.__file__, str)
    assert copied.read_bytes() == Path(gallery.__file__).read_bytes()


@pytest.mark.fast_always
def test_native_ci_installs_and_verifies_pinned_veris() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/python-package.yml").read_text())
    quality = workflow["jobs"]["quality"]
    commands = "\n".join(step.get("run", "") for step in quality["steps"])
    assert ".[dev,jcm,veros,veris]" in commands
    assert quality["env"]["VERIS_COMMIT"] == "ee9614eb51726509c2b7f037a2a20b25c8d117e5"
    assert 'distribution("veris")' in commands and "VERIS_COMMIT" in commands
    assert "not veros_autodiff" in commands
    assert (
        workflow["jobs"]["veros-autodiff"]["env"]["VEROS_AD_COMMIT"]
        == "7a8c964cf00b5aa0713c995edd760643a431b3c9"
    )


@pytest.mark.slow
@pytest.mark.parametrize("explicit_ice", [False, True])
def test_real_global_native_gallery(tmp_path: Path, explicit_ice: bool) -> None:
    pytest.importorskip("veris")
    pytest.importorskip("veros")
    code = """
import json, sys
from dataclasses import fields
from pathlib import Path
import numpy as np
import xarray as xr
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import VerisConfig
from vercor.setups.gallery.run_veris_dataveros_era5 import _make_coupler
explicit = sys.argv[2] == 'True'
config = None
if explicit:
    polar = np.broadcast_to((np.abs(np.arange(-78, 80, 4)) > 60)[None, :], (90, 40))
    config = VerisConfig(
        initial_conditions={'Area': 0.5 * polar, 'hIceMean': 0.25 * polar, 'TSurf': 260.0},
        output=OutputSpec(period=PeriodOutput(frequency='day', variables=('Area', 'hIceMean', 'TSurf'))),
    )
c = _make_coupler(loglevel='error', float_type='float64', steps=2 if explicit else 1, veris_config=config)
initial = c.initial_state()
before = initial._component_state('CICE').payload.state
final = c.run(initial, output=OutputTarget(sys.argv[1]))
payload = final._component_state('CICE').payload
state = payload.state
area = np.asarray(final.component('CICE').field('ice_fraction'))
wet = np.asarray(final.component('OCN').grid.binary_mask) > 0
assert area.shape == (40, 90) and ((area >= 0) & (area <= 1)).all()
for item in fields(state):
    value = np.asarray(getattr(state, item.name))[2:-2, 2:-2]
    assert np.isfinite(value[wet.T]).all(), item.name
for name in ('Area', 'hIceMean', 'hSnowMean'):
    value = np.asarray(getattr(state, name))[2:-2, 2:-2].T
    assert (value[~wet] == 0).all(), name
if explicit:
    assert np.max(np.abs(np.asarray(state.Area - before.Area))) > 0
    assert np.max(np.abs(np.asarray(state.hIceMean - before.hIceMean))) > 0
else:
    assert (np.asarray(before.Area) == 0).all()
component = c.components['CICE']
assert component.spec.lifecycle.setup.__self__.resources.model_substeps == 24
output = Path(sys.argv[1])
if explicit:
    assert len(tuple(output.glob('cice.averages.*.nc'))) == 2
else:
    assert not tuple(output.glob('cice.averages.*.nc'))  # no monthly boundary after one day
with xr.open_dataset(output / 'cice.snapshot.nc') as ds:
    np.testing.assert_allclose(ds.Area[0], area)
    assert ds.forc_salt_surface.attrs['units'] == 'g kg-1 m s-1'
assert (output / 'cice.runtime_fields.nc').is_file()
print(json.dumps({
    'wet': int(wet.sum()), 'ice_cells': int((area > 0).sum()),
    'area_min': float(area.min()), 'area_max': float(area.max()), 'steps': c.clock.steps,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(explicit_ice)],
        capture_output=True,
        text=True,
    )
    log = tmp_path / "native-run.log"
    log.write_text(result.stdout + result.stderr)
    assert (
        result.returncode == 0
    ), f"ERROR native gallery failed; see {log}: {result.stderr[-700:]}"
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence["wet"] > 0 and evidence["steps"] == (2 if explicit_ice else 1)
