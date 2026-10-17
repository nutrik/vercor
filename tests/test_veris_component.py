"""Exercise standalone Veris configuration, lazy loading and resource ownership."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import DataComponent, StepContext
from vercor.dtypes import DTypePolicy
from vercor.regridding import bilinear
from tests.test_shared_surface_fluxes import surface_fields


def small_ocean() -> tuple[RectilinearGrid, Any]:
    """Return non-square native physical geometry with one dry tracer cell."""
    veris = pytest.importorskip("veris")
    jax.config.update("jax_enable_x64", True)
    mask = np.ones((3, 2))
    mask[0, 0] = 0.0
    mask_u = mask * np.roll(mask, 1, axis=0)
    mask_v = mask * np.roll(mask, 1, axis=1)
    mask_v[:, 0] = 0.0
    grid = RectilinearGrid(
        "small-ocean",
        longitude=np.arange(3.0),
        latitude=np.array([-1.0, 1.0]),
        longitude_edges=np.arange(4.0) - 0.5,
        latitude_edges=np.array([-2.0, 0.0, 2.0]),
        binary_mask=mask.T,
    )
    geometry = veris.OceanGeometry(
        maskT=mask[..., None],
        maskU=mask_u[..., None],
        maskV=mask_v[..., None],
        ht=-1000 * mask,
        coriolis_t=np.full((3, 2), 1e-4),
        dxt=np.full(3, 2e4),
        dxu=np.full(3, 2e4),
        dyt=np.full(2, 2e4),
        dyu=np.full(2, 2e4),
        area_t=np.full((3, 2), 4e8),
        area_u=np.full((3, 2), 4e8),
        area_v=np.full((3, 2), 4e8),
    )
    return grid, geometry


def cold_fields() -> dict[str, jax.Array]:
    """Return smooth freezing conditions with separate prescribed water phases."""
    result = surface_fields()
    for name, value in {
        "temperature": 260.0,
        "potential_temperature": 260.0,
        "specific_humidity": 0.001,
        "density": 1.3,
        "sea_surface_temperature": 271.15,
        "ocean_u_velocity": 0.0,
        "net_shortwave_radiation_flux": 20.0,
        "downward_shortwave_radiation_flux": 30.0,
        "downward_longwave_radiation_flux": 180.0,
        "rain_rate": 0.0,
        "snow_rate": 1e-6,
    }.items():
        result[name] = jnp.full((2, 3), value)
    return result


def small_coupler(
    *, dt: float = 600.0, steps: int = 2, config: Any = None, enable_x64: bool = True
) -> tuple[Coupler, Any, Any, Any]:
    """Assemble real Veris with a prescribed callable-runtime forcing source."""
    from vercor.setups import VerisConfig, make_veris
    from vercor.recipes import (
        ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
        OCEAN_TO_VERIS_FORCING_FIELDS,
    )

    grid, geometry = small_ocean()
    config = (
        VerisConfig(
            model_timestep=timedelta(seconds=300),
            custom_parameters={"nEVPsteps": 8},
            initial_conditions={"Area": 0.5, "hIceMean": 0.25, "TSurf": 260.0},
        )
        if config is None
        else config
    )
    component = make_veris(grid, geometry, config=config)
    forcing = DataComponent("FORCING", grid, cold_fields())
    route = Exchange(
        "FORCING",
        config.name,
        (*ATMOSPHERE_TO_VERIS_FORCING_FIELDS, *OCEAN_TO_VERIS_FORCING_FIELDS),
        quantity="intensive",
        regridder_factory=bilinear,
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt, steps),
        components=(forcing, component),
        exchanges=(route,),
        run_order=("FORCING", config.name),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=enable_x64)),
    )
    return coupler, component, geometry, config


@pytest.mark.fast_always
def test_core_and_veris_configuration_remain_lazy_without_models() -> None:
    code = """
import builtins, json, sys
original = builtins.__import__
def block(name, *args, **kwargs):
    if name.split('.')[0] in {'veris', 'veros'}:
        raise ImportError('optional model blocked')
    return original(name, *args, **kwargs)
builtins.__import__ = block
import vercor
from vercor.setups import VerisConfig, VerisVerosDataSetup, make_veris, make_veris_verosdata
assert VerisConfig().name == 'CICE'
assert not {'veris', 'veros'} & sys.modules.keys()
print(json.dumps({'lazy': True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout)["lazy"]


@pytest.mark.fast_always
def test_configuration_is_frozen_and_owns_initial_arrays() -> None:
    from vercor.setups import VerisConfig

    initial = np.full((3, 2), 0.5)
    parameters: dict[str, object] = {"nEVPsteps": 8}
    config = VerisConfig(
        initial_conditions={"Area": initial}, custom_parameters=parameters
    )
    initial[:] = 0.0
    parameters["nEVPsteps"] = 99
    assert config.initial_conditions is not None
    assert config.custom_parameters is not None
    np.testing.assert_array_equal(config.initial_conditions["Area"], 0.5)
    assert config.custom_parameters["nEVPsteps"] == 8
    with pytest.raises(FrozenInstanceError):
        setattr(config, "name", "other")


@pytest.mark.parametrize("seconds", [0, -1])
def test_native_timestep_must_be_positive(seconds: int) -> None:
    from vercor.setups import VerisConfig

    with pytest.raises(ValueError, match="positive"):
        VerisConfig(model_timestep=timedelta(seconds=seconds))


@pytest.mark.parametrize(
    "options,error",
    [
        ({"name": ""}, ValueError),
        ({"model_timestep": 300}, TypeError),
        ({"output": object()}, TypeError),
    ],
)
def test_configuration_rejects_invalid_controls(
    options: dict[str, Any], error: type[Exception]
) -> None:
    from vercor.setups import VerisConfig

    with pytest.raises(error):
        VerisConfig(**options)


@pytest.mark.parametrize("incomplete", [False, True])
def test_factory_reports_missing_standalone_capability(
    monkeypatch: pytest.MonkeyPatch, incomplete: bool
) -> None:
    from vercor.setups import make_veris

    grid, _ = small_ocean()
    monkeypatch.setitem(
        sys.modules, "veris", ModuleType("veris") if incomplete else None
    )
    with pytest.raises(ImportError, match=r"vercor\[veris\]"):
        make_veris(grid, object())


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("dtype", "float32"),
        ("nx", 99),
        ("ny", 99),
        ("use_sharding", True),
        ("deltatDyn", 1.0),
        ("deltatTherm", 1.0),
        ("enable_cyclic_y", True),
    ],
)
def test_runtime_owned_native_parameters_reject_conflicts(
    parameter: str, value: object
) -> None:
    from vercor.setups import VerisConfig

    config = VerisConfig(
        model_timestep=timedelta(seconds=300), custom_parameters={parameter: value}
    )
    with pytest.raises(ValueError, match=parameter):
        coupler, *_ = small_coupler(config=config)
        coupler.initial_state()


@pytest.mark.parametrize("dt", [0.0, 450.0])
def test_coupling_interval_must_be_positive_and_divisible(dt: float) -> None:
    with pytest.raises(ValueError, match="positive|divide"):
        coupler, *_ = small_coupler(dt=dt)
        coupler.initial_state()


def test_geometry_requires_exact_ocean_grid_and_binary_mask() -> None:
    from vercor.setups import make_veris

    grid, geometry = small_ocean()
    with pytest.raises(TypeError, match="OceanGeometry"):
        make_veris(grid, object())
    other = RectilinearGrid(
        "other",
        longitude=np.arange(4.0),
        latitude=np.array([-1.0, 1.0]),
        binary_mask=np.ones((2, 4)),
    )
    with pytest.raises(ValueError, match="grid|shape"):
        make_veris(other, geometry)
    landless = RectilinearGrid(
        "other",
        longitude=np.arange(3.0),
        latitude=np.array([-1.0, 1.0]),
        binary_mask=np.ones((2, 3)),
    )
    with pytest.raises(ValueError, match="mask"):
        make_veris(landless, geometry)


def test_paired_factory_initializes_once_and_shares_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import fields
    import vercor.setups._veris as paired
    from vercor.setups._external.veros_data import VerosDataSnapshot

    grid, geometry = small_ocean()
    data = {
        name: np.broadcast_to(np.asarray(value), (12, *grid.shape))
        for name, value in cold_fields().items()
        if name.startswith("sea_surface") or name.startswith("ocean_")
    }
    snapshot = VerosDataSnapshot(
        grid,
        {field.name: getattr(geometry, field.name) for field in fields(geometry)},
        data,
    )
    calls: list[bool] = []

    def load() -> Any:
        calls.append(True)
        return snapshot

    monkeypatch.setattr(paired, "load_veros_data_snapshot", load)
    result = paired.make_veris_verosdata()
    assert result.seaice.grid is result.ocean.grid is grid
    assert isinstance(result.ocean, DataComponent)
    assert calls == [True]
    with pytest.raises(ValueError, match="distinct"):
        paired.make_veris_verosdata(ocean_name="CICE")
    assert calls == [True]


def test_optional_dependency_is_immutable_and_preserves_veros_extras() -> None:
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]
    assert extras["veris"] == [
        "veris @ git+https://github.com/team-ocean/veris.git@4551fc86181c660ceb0010160ec7f569d97e52c6"
    ]
    assert extras["veros"] == ["veros>=1.6.2,<1.7"]


@pytest.mark.parametrize("enable_x64", [False, True])
def test_native_state_and_diagnostics_follow_runtime_dtype(enable_x64: bool) -> None:
    coupler, component, _, _ = small_coupler(steps=1, enable_x64=enable_x64)
    initial = coupler.initial_state()
    final = coupler.run(initial)
    payload = final._component_state("CICE").payload
    assert payload is not None
    expected = jnp.float64 if enable_x64 else jnp.float32
    assert all(
        value.dtype == expected
        for value in jax.tree_util.tree_leaves(
            (
                payload.state,
                payload.diagnostics,
                payload.physical_constants,
                payload.interval_budget,
            )
        )
    )
    assert payload.checked_native_steps.dtype == jnp.int32
    assert int(payload.checked_native_steps) == 2
    owner = component.spec.lifecycle.setup.__self__
    assert owner.resources.configuration.dtype == (
        "float64" if enable_x64 else "float32"
    )
    assert (
        owner.resources.configuration.deltatDyn
        == owner.resources.configuration.deltatTherm
        == 300.0
    )
    assert owner.resources.model_substeps == 2


def test_setup_owns_geometry_and_resources_remain_fixed() -> None:
    from vercor.exceptions import ComponentError

    coupler, component, geometry, _ = small_coupler()
    geometry.maskT[:] = 0.0
    initial = coupler.initial_state()
    payload = initial._component_state("CICE").payload
    assert payload is not None
    np.testing.assert_array_equal(
        payload.state.iceMask[2:-2, 2:-2].T, component.grid.binary_mask
    )
    owner = component.spec.lifecycle.setup.__self__
    resources = owner.resources
    with pytest.raises(FrozenInstanceError):
        setattr(resources, "model_substeps", 99)
    other = Coupler(
        Clock(datetime(2000, 1, 1), 600.0, 1),
        components=(component,),
        run_order=("CICE",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )
    other.initial_state()
    assert owner.resources is resources

    incompatible = Coupler(
        Clock(datetime(2000, 1, 1), 300.0, 1),
        components=(component,),
        run_order=("CICE",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )
    with pytest.raises(ComponentError, match="new Veris component"):
        incompatible.initial_state()
    assert owner.resources is resources


def test_step_requires_setup_and_native_payload() -> None:
    from vercor.exceptions import ComponentError

    coupler, component, _, _ = small_coupler()
    context = StepContext(600.0, dtype=DTypePolicy(enable_x64=True))
    with pytest.raises(ComponentError, match="initialized"):
        component.step(cold_fields(), context, None)
    coupler.initial_state()
    with pytest.raises(ComponentError, match="payload"):
        component.step(cold_fields(), context, None)
