"""Check one-owner ocean/ice assembly and prognostic gallery contracts."""

from __future__ import annotations

from datetime import timedelta
import importlib
from typing import Any

import numpy as np
import pytest

from tests._veros_surface_support import surface_setup_class
from vercor.setups import VerisConfig, VerosConfig

pytestmark = pytest.mark.fast_always


def test_prognostic_pair_initializes_one_ocean_on_one_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.setups import make_veris_veros
    from vercor.setups._external import veros_gcm_state

    setup = surface_setup_class()
    calls: list[Any] = []
    original = setup.setup

    def initialize(self: Any) -> None:
        calls.append(self)
        original(self)

    monkeypatch.setattr(setup, "setup", initialize)
    monkeypatch.setattr(
        veros_gcm_state, "_select_veros_setup_class", lambda *args: setup
    )
    pair = make_veris_veros(config=VerisConfig(model_timestep=timedelta(seconds=300)))
    assert len(calls) == 1
    assert pair.seaice.grid is pair.ocean.grid
    grid = pair.ocean.grid
    vs = calls[0].state.variables
    np.testing.assert_array_equal(grid.longitude_edges, vs.xu[1:-2])
    np.testing.assert_array_equal(grid.latitude_edges, vs.yu[1:-2])
    ice_owner = pair.seaice.spec.lifecycle.setup.__self__
    np.testing.assert_array_equal(ice_owner.geometry.area_t, vs.area_t[2:-2, 2:-2])


@pytest.mark.parametrize("config", [VerosConfig(), VerosConfig(setup="acc")])
def test_pair_rejects_incompatible_ocean_before_initialization(
    config: VerosConfig,
) -> None:
    from vercor.setups import make_veris_veros

    with pytest.raises(ValueError, match="surface_fluxes|global"):
        make_veris_veros(ocean_config=config)


def test_packaged_prognostic_example_exists_with_standard_contract() -> None:
    module = importlib.import_module("vercor.setups.gallery.run_veris_veros_era5")
    assert callable(module.run_setup) and callable(module._make_coupler)


def test_prognostic_gallery_assembles_bidirectional_total_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_veris_component import small_ocean
    from tests.test_shared_surface_fluxes import surface_fields
    from vercor.components import (
        CallableComponent,
        ComponentSpec,
        ComponentMaskSpec,
        DataComponent,
    )
    from vercor.fields import _flatten_field_items
    from vercor.recipes import (
        OBSERVED_10M_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS,
        OCEAN_TO_SEAICE_FORCING_FIELDS,
        SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS,
    )
    from vercor.regridding import conservative

    module = importlib.import_module("vercor.setups.gallery.run_veris_veros_era5")
    grid, _ = small_ocean()
    fields = surface_fields()
    ice = CallableComponent(
        "CICE",
        grid,
        lambda f: {},
        spec=ComponentSpec(
            inputs=tuple(fields),
            outputs=(
                "ice_fraction",
                *_flatten_field_items(SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS),
            ),
            initial_fields={
                "ice_fraction": 0.0,
                **{
                    name: 0.0
                    for name in _flatten_field_items(SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS)
                },
            },
            masks=ComponentMaskSpec(fractional_field="ice_fraction"),
        ),
    )
    ocean = DataComponent(
        "OCN",
        grid,
        {
            name: fields[name]
            for name in _flatten_field_items(OCEAN_TO_SEAICE_FORCING_FIELDS)
        },
    )
    atmosphere = DataComponent(
        "ATM",
        grid,
        {
            name: fields[name]
            for name in _flatten_field_items(
                OBSERVED_10M_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS
            )
        },
    )
    land = DataComponent("LND", grid, {"land_surface_temperature": 273.15})
    monkeypatch.setattr(
        module,
        "make_veris_veros",
        lambda **kwargs: type("Pair", (), {"seaice": ice, "ocean": ocean})(),
    )
    monkeypatch.setattr(module, "make_era5_atmosphere", lambda: atmosphere)
    monkeypatch.setattr(module, "make_era5_land", lambda: land)
    coupler = module._make_coupler(loglevel="error", float_type="float64", steps=2)
    assert tuple(coupler.run_order) == ("LND", "CICE", "OCN", "ATM")
    assert [
        (route.source, route.target, route.quantity, route.source_coverage)
        for route in coupler.exchanges
    ] == [
        ("ATM", "CICE", "intensive", "component"),
        ("OCN", "CICE", "intensive", "component"),
        ("CICE", "OCN", "source_flux", "binary"),
        ("CICE", "OCN", "source_flux", "binary"),
    ]
    assert coupler.exchanges[-2].fields == SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS[:-1]
    assert coupler.exchanges[-1].fields == SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS[-1:]
    assert coupler.exchanges[-2].regridder_factory is conservative
    assert coupler.exchanges[-2].route_id != coupler.exchanges[-1].route_id


def test_new_templates_are_lazy_and_copyable(tmp_path: Any) -> None:
    import subprocess
    import sys
    from pathlib import Path
    from click.testing import CliRunner
    from vercor.cli import cli

    result = CliRunner().invoke(cli, ["show-setups"])
    assert result.exit_code == 0
    for stem in ("run_veris_calibration", "run_veris_veros_era5"):
        assert stem in result.output
        copied = CliRunner().invoke(cli, ["copy-setup", stem, "--to", str(tmp_path)])
        assert copied.exit_code == 0, copied.output
        module = importlib.import_module(f"vercor.setups.gallery.{stem}")
        assert isinstance(module.__file__, str)
        assert (Path(tmp_path) / f"{stem}.py").read_bytes() == Path(
            module.__file__
        ).read_bytes()
    code = """
import builtins, sys
original = builtins.__import__
def block(name, *args, **kwargs):
    if name.split('.')[0] in {'veris', 'veros'}:
        raise ImportError('optional model blocked')
    return original(name, *args, **kwargs)
builtins.__import__ = block
from vercor.setups import VerisVerosSetup, make_veris_veros
from vercor.setups.gallery import run_veris_calibration, run_veris_veros_era5
assert not {'veris', 'veros'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def _prognostic_coupler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    execution: Any = "host",
    steps: int = 1,
    export_open_water_surface: bool = False,
) -> Any:
    """Use the public pair with real native small-grid ocean and prescribed air."""
    from datetime import datetime
    from vercor import Clock, Coupler, Exchange, RuntimeOptions
    from vercor.components import DataComponent
    from vercor.dtypes import DTypePolicy
    from vercor.fields import _flatten_field_items
    from vercor.recipes import (
        OBSERVED_10M_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS,
        OCEAN_TO_SEAICE_FORCING_FIELDS,
        SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS,
    )
    from vercor.regridding import conservative
    from vercor.setups import make_veris_veros
    from vercor.setups._external import veros_gcm_state
    from vercor.setups.gallery.run_veris_calibration import _cold_fields

    setup = surface_setup_class(execution=execution)
    monkeypatch.setattr(
        veros_gcm_state, "_select_veros_setup_class", lambda *args: setup
    )
    pair = make_veris_veros(
        config=VerisConfig(
            model_timestep=timedelta(seconds=300),
            custom_parameters={"nEVPsteps": 8},
            initial_conditions={"Area": 0.5, "hIceMean": 0.25, "TSurf": 260.0},
        ),
        ocean_config=VerosConfig(
            execution=execution,
            surface_forcing="surface_fluxes",
            custom_parameters={"dt_mom": 600.0},
            export_open_water_surface=export_open_water_surface,
        ),
    )
    fields = _cold_fields(
        pair.ocean.grid.shape, policy=DTypePolicy(enable_x64=True), calm=True
    )
    atmosphere = DataComponent(
        "ATM",
        pair.ocean.grid,
        {
            name: fields[name]
            for name in _flatten_field_items(
                OBSERVED_10M_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS
            )
        },
    )
    return Coupler(
        Clock(datetime(2000, 1, 1), 600.0, steps),
        components=(atmosphere, pair.seaice, pair.ocean),
        exchanges=(
            Exchange(
                "ATM",
                "CICE",
                OBSERVED_10M_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS,
                quantity="intensive",
            ),
            Exchange(
                "OCN", "CICE", OCEAN_TO_SEAICE_FORCING_FIELDS, quantity="intensive"
            ),
            Exchange(
                "CICE",
                "OCN",
                SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS[:-1],
                route_id="CICE->OCN.scalars",
                quantity="source_flux",
                source_coverage="binary",
                regridder_factory=conservative,
            ),
            Exchange(
                "CICE",
                "OCN",
                SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS[-1:],
                route_id="CICE->OCN.stress",
                quantity="source_flux",
                source_coverage="binary",
            ),
            *(
                (
                    Exchange(
                        "CICE",
                        "OCN",
                        ("ice_fraction",),
                        route_id="CICE->OCN.coverage",
                        quantity="intensive",
                        source_coverage="binary",
                    ),
                )
                if export_open_water_surface
                else ()
            ),
        ),
        run_order=("CICE", "OCN", "ATM"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
        log_level="error",
    )


def test_actual_prognostic_pair_closes_native_heat_and_virtual_salt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax
    from tests.test_veris_rollouts import native_payload

    coupler = _prognostic_coupler(monkeypatch)
    initial = coupler.initial_state()
    final = coupler.run(initial)
    before = initial._component_state("OCN").payload
    after = final._component_state("OCN").payload
    assert before is not None and after is not None
    ice_before, ice_after = native_payload(initial), native_payload(final)
    fields = final.component("CICE")
    area = np.asarray(before.variables.area_t)[2:-2, 2:-2].T
    depth = np.asarray(before.variables.dzt)
    changes = {}
    for name in ("temp", "salt"):
        first = np.asarray(getattr(before.variables, name))[
            2:-2, 2:-2, :, before.variables.tau
        ].transpose(1, 0, 2)
        last = np.asarray(getattr(after.variables, name))[
            2:-2, 2:-2, :, after.variables.tau
        ].transpose(1, 0, 2)
        changes[name] = np.sum((last - first) * area[..., None] * depth)
    heat = 600 * np.sum(np.asarray(fields.field("sea_surface_heat_flux")) * area)
    mass = np.asarray(fields.field("sea_surface_mass_flux"))
    salt = np.asarray(fields.field("sea_surface_salt_mass_flux"))
    virtual = 600 * np.sum((-34.7 * mass + 1000 * salt) / 1000 * area)
    heat_change = changes["temp"] * after.settings.rho_0 * 3991.86795711963
    for actual, expected in ((heat_change, heat), (changes["salt"], virtual)):
        assert abs(actual - expected) / max(1.0, abs(expected)) < 1e-10
    p = ice_after.physical_constants
    solid_change = (
        p.rhoIce * (ice_after.state.hIceMean - ice_before.state.hIceMean)
        + p.rhoSnow * (ice_after.state.hSnowMean - ice_before.state.hSnowMean)
    )[2:-2, 2:-2].T
    residual = np.sum(
        (solid_change + 600 * mass - 600 * ice_after.interval_budget.external_mass_flux)
        * area
    )
    scale = max(1.0, abs(float(np.sum(solid_change * area))))
    assert abs(float(residual)) / scale < 1e-10
    assert all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(ice_after)
    )
    assert (
        np.max(
            abs(
                np.asarray(final.component("OCN").field("sea_surface_temperature"))
                - np.asarray(initial.component("OCN").field("sea_surface_temperature"))
            )
        )
        > 1e-8
    )


def test_prognostic_run_setup_delegates_cli_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.output import OutputTarget

    module = importlib.import_module("vercor.setups.gallery.run_veris_veros_era5")
    captured: dict[str, Any] = {}

    class Runner:
        def run(self, *, output: OutputTarget) -> None:
            captured["output"] = output

    def build(**kwargs: Any) -> Runner:
        captured.update(kwargs)
        return Runner()

    monkeypatch.setattr(module, "_make_coupler", build)
    module.run_setup(loglevel="warning", float_type="float32")
    assert captured["loglevel"] == "warning"
    assert captured["float_type"] == "float32"
    assert str(captured["output"].directory) == "."
