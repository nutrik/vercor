"""Protect public four-component assembly, route ownership and copyability."""

from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any
import subprocess
import sys

import numpy as np
import pytest
from click.testing import CliRunner
from vercor import Clock
from vercor.cli import cli
from vercor.dtypes import DTypePolicy
from vercor.setups import (
    JCMAtmosphereConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerisConfig,
    VerosConfig,
)

pytestmark = pytest.mark.fast_always


def native_coupler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    execution: Any = "host",
    steps: int = 2,
    output: bool = False,
) -> Any:
    """Use actual 4x4 native ocean/ice and flat T21 JCM; reduced EVP test numerics."""
    from tests._veros_surface_support import surface_setup_class
    from tests.test_jcm_surface_forcing import native_case
    from vercor.setups import JCMInputs
    from vercor.setups._external import veros_gcm_state
    from vercor.output import OutputSpec, PeriodOutput

    DTypePolicy(enable_x64=True)._ensure_jax_capability()
    setup = surface_setup_class(execution=execution, dt=1800.0)
    calls: list[Any] = []
    initialize = setup.setup

    def counted_setup(self: Any) -> None:
        calls.append(self)
        initialize(self)

    monkeypatch.setattr(setup, "setup", counted_setup)
    monkeypatch.setattr(veros_gcm_state, "_select_veros_setup_class", lambda *a: setup)
    coords, terrain, _, forcing = native_case(resolution=21)
    import jax.numpy as jnp

    shape = terrain.fmask.shape
    forcing = forcing.copy(
        stl_am=jnp.full(shape, 278.0),
        sea_surface_temperature=jnp.full(shape, 271.0),
        soilw_am=jnp.full(shape, 0.5),
        snowc_am=jnp.zeros(shape),
    )
    period = (
        OutputSpec(period=PeriodOutput(frequency="step")) if output else OutputSpec()
    )
    coupler = import_module("vercor.setups.gallery.run_veris_veros_jcm").build_coupler(
        jcm_inputs=JCMInputs(coords, terrain, forcing),
        clock=Clock(datetime(2000, 1, 3, 12), 1800.0, steps, calendar="gregorian"),
        veris_config=VerisConfig(
            atmosphere_forcing="model_level",
            model_timestep=timedelta(seconds=300),
            custom_parameters={"nEVPsteps": 8},
            initial_conditions={"Area": 0.5, "hIceMean": 0.25, "TSurf": 260.0},
            output=period,
        ),
        ocean_config=VerosConfig(
            execution=execution,
            surface_forcing="surface_fluxes",
            export_open_water_surface=True,
            output=period,
        ),
        jcm_config=JCMLandAtmosphereConfig(
            atmosphere=JCMAtmosphereConfig(
                seaice_coupling=True,
                spinup=Spinup(enabled=False),
                save_interval=timedelta(seconds=1800),
                output=period,
            )
        ),
        log_level="error",
        dtype=DTypePolicy(enable_x64=True),
    )

    assert len(calls) == 1
    return coupler


def test_gallery_routes_preserve_surface_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor import recipes as r
    from vercor.regridding import bilinear as b, conservative as c

    coupler = native_coupler(monkeypatch, steps=0)
    assert tuple(coupler.run_order) == ("LND", "CICE", "OCN", "ATM")
    expected = [
        (
            "ATM",
            "CICE",
            r.MODEL_LEVEL_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS,
            "intensive",
            "component",
            b,
        ),
        ("OCN", "CICE", r.OCEAN_TO_SEAICE_FORCING_FIELDS, "intensive", "binary", b),
        (
            "CICE",
            "OCN",
            r.SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS[:-1],
            "source_flux",
            "binary",
            c,
        ),
        (
            "CICE",
            "OCN",
            r.SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS[-1:],
            "source_flux",
            "binary",
            b,
        ),
        ("CICE", "OCN", r.SEAICE_COVERAGE_FIELDS, "intensive", "binary", b),
        (
            "OCN",
            "ATM",
            r.OCEAN_TO_ATMOSPHERE_OPEN_WATER_FIELDS,
            "source_flux",
            "binary",
            c,
        ),
        (
            "CICE",
            "ATM",
            r.SEAICE_TO_ATMOSPHERE_SURFACE_FIELDS,
            "source_flux",
            "component",
            c,
        ),
        ("CICE", "ATM", r.SEAICE_COVERAGE_FIELDS, "source_flux", "binary", c),
        ("LND", "ATM", r.LAND_TO_ATMOSPHERE_STATE_FIELDS, "intensive", "component", b),
    ]
    assert [
        (
            x.source,
            x.target,
            x.fields,
            x.quantity,
            x.source_coverage,
            x.regridder_factory,
        )
        for x in coupler.exchanges
    ] == expected
    assert len({x.route_id for x in coupler.exchanges}) == 9
    assert coupler.components["CICE"].grid is coupler.components["OCN"].grid
    initial = (
        coupler.initial_state()
    )  # Includes SurfaceMaskPolicy with explicit quantities.
    np.testing.assert_allclose(
        initial.component("CICE").field("ice_fraction"),
        np.where(coupler.components["CICE"].grid.binary_mask, 0.5, 0),
    )


def test_cli_copy_and_lazy_optional_imports(tmp_path: Path) -> None:
    assert "run_veris_veros_jcm" in CliRunner().invoke(cli, ["show-setups"]).output
    result = CliRunner().invoke(
        cli, ["copy-setup", "run_veris_veros_jcm", "--to", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    module = import_module("vercor.setups.gallery.run_veris_veros_jcm")
    assert isinstance(module.__file__, str)
    assert (tmp_path / "run_veris_veros_jcm.py").read_bytes() == Path(
        module.__file__
    ).read_bytes()
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins, sys
original=builtins.__import__
def block(name,*a,**k):
    if name.split('.')[0] in {'jcm','veris','veros'}: raise ImportError('optional blocked')
    return original(name,*a,**k)
builtins.__import__=block
from vercor.setups.gallery.run_veris_veros_jcm import build_coupler, run_setup
assert not {'jcm','veris','veros'} & sys.modules.keys()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clock": Clock(datetime(2000, 1, 3), 86400, 1, calendar="noleap")},
        {"veris_config": VerisConfig()},
        {"ocean_config": VerosConfig(surface_forcing="surface_fluxes")},
        {"jcm_config": JCMLandAtmosphereConfig()},
    ],
)
def test_incompatible_explicit_modes_are_rejected_before_models(
    kwargs: dict[str, Any],
) -> None:
    module = import_module("vercor.setups.gallery.run_veris_veros_jcm")
    with pytest.raises(
        ValueError, match="Gregorian|model_level|export|seaice_coupling"
    ):
        module.build_coupler(**kwargs)


def test_precision_is_selected_before_native_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax

    module = import_module("vercor.setups.gallery.run_veris_veros_jcm")

    class FactoryReached(Exception):
        pass

    def factory(**kwargs: Any) -> Any:
        assert jax.config.jax_enable_x64
        raise FactoryReached

    monkeypatch.setattr(module, "make_veris_veros", factory)
    previous = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(FactoryReached):
            module.build_coupler(dtype=DTypePolicy(enable_x64=True))
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_injected_pair_cannot_receive_overrides() -> None:
    from types import SimpleNamespace

    module = import_module("vercor.setups.gallery.run_veris_veros_jcm")
    with pytest.raises(ValueError, match="pair.*overrides"):
        module.build_coupler(
            pair=SimpleNamespace(),
            veris_config=VerisConfig(atmosphere_forcing="model_level"),
        )
