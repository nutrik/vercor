"""Verify copied Veros geometry and time-selected prescribed ocean surfaces."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vercor import Clock, Coupler, Exchange
from vercor.components import CallableComponent, ComponentSpec
from vercor.regridding import bilinear

pytestmark = pytest.mark.fast_always


@pytest.fixture
def native() -> Any:
    shape = (7, 6)
    values: dict[str, Any] = {}
    for index, name in enumerate(("ht", "coriolis_t", "area_t", "area_u", "area_v")):
        values[name] = np.arange(42, dtype=float).reshape(shape) + 100 * (index + 1)
    for index, name in enumerate(("maskT", "maskU", "maskV")):
        values[name] = np.ones((*shape, 2))
        values[name][2 + index, 2, :] = 0.0
    for name, size in (("dxt", 7), ("dxu", 7), ("dyt", 6), ("dyu", 6)):
        values[name] = np.arange(size, dtype=float) + 1000
    values.update(
        xt=np.arange(7) - 1.5,
        xu=np.arange(7) - 1.0,
        yt=np.arange(6) - 1.5,
        yu=np.arange(6) - 1.0,
    )
    values["sst_clim"] = np.broadcast_to(
        np.arange(12, dtype=float), (*shape, 12)
    ).copy()
    values["sss_clim"] = values["sst_clim"] + 30.0
    return SimpleNamespace(variables=SimpleNamespace(**values))


def test_snapshot_preserves_geometry_axes_staggering_and_units(native: Any) -> None:
    from vercor.setups._external.veros_data import snapshot_veros_state

    snapshot = snapshot_veros_state(native)
    assert snapshot.grid.shape == (2, 3)
    for name in (
        "maskT",
        "ht",
        "coriolis_t",
        "area_t",
    ):
        expected = getattr(native.variables, name)[2:-2, 2:-2]
        np.testing.assert_array_equal(
            snapshot.geometry[name], -expected if name == "ht" else expected
        )
    # Native Veris owns west/south faces; Veros stores east/north faces.
    for name in ("maskU", "area_u"):
        np.testing.assert_array_equal(
            snapshot.geometry[name], getattr(native.variables, name)[1:-3, 2:-2]
        )
    for name in ("maskV", "area_v"):
        np.testing.assert_array_equal(
            snapshot.geometry[name], getattr(native.variables, name)[2:-2, 1:-3]
        )
    for name in ("dxt", "dyt"):
        np.testing.assert_array_equal(
            snapshot.geometry[name], getattr(native.variables, name)[2:-2]
        )
    for name in ("dxu", "dyu"):
        np.testing.assert_array_equal(
            snapshot.geometry[name], getattr(native.variables, name)[1:-3]
        )
    np.testing.assert_array_equal(
        snapshot.grid.binary_mask, native.variables.maskT[2:-2, 2:-2, -1].T
    )
    np.testing.assert_array_equal(snapshot.grid.longitude, native.variables.xt[2:-2])
    np.testing.assert_array_equal(snapshot.grid.latitude, native.variables.yt[2:-2])
    np.testing.assert_array_equal(
        snapshot.grid.longitude_edges, native.variables.xu[1:-2]
    )
    np.testing.assert_array_equal(
        snapshot.grid.latitude_edges, native.variables.yu[1:-2]
    )
    np.testing.assert_allclose(snapshot.fields["sea_surface_temperature"][0], 273.15)
    np.testing.assert_allclose(snapshot.fields["sea_surface_salinity"][0], 30.0)
    for name in ("ocean_u_velocity", "ocean_v_velocity", "sea_surface_height"):
        assert snapshot.fields[name].shape == (12, 2, 3)
        np.testing.assert_array_equal(snapshot.fields[name], 0.0)


def test_snapshot_owns_read_only_arrays_without_native_owner(native: Any) -> None:
    from vercor.setups._external.veros_data import snapshot_veros_state

    snapshot = snapshot_veros_state(native)
    original = np.array(snapshot.geometry["area_t"])
    native.variables.area_t[:] = -999.0
    native.variables.sst_clim[:] = -999.0
    np.testing.assert_array_equal(snapshot.geometry["area_t"], original)
    np.testing.assert_allclose(snapshot.fields["sea_surface_temperature"][0], 273.15)
    assert tuple(field.name for field in fields(snapshot)) == (
        "grid",
        "geometry",
        "fields",
    )
    with pytest.raises(TypeError):
        snapshot.geometry["ht"] = np.zeros((3, 2))  # type: ignore[index]
    assert not np.asarray(snapshot.geometry["area_t"]).flags.writeable


def test_loader_initializes_once_and_never_advances(
    native: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vercor.setups._external import veros_data

    calls: list[str] = []

    def initialize() -> Any:
        calls.append("setup")
        return native

    monkeypatch.setattr(veros_data, "_initialize_veros", initialize)
    snapshot = veros_data.load_veros_data_snapshot()
    assert snapshot.grid.shape == (2, 3)
    assert calls == ["setup"]


def test_data_ocean_reuses_grid_and_monthly_selection(native: Any) -> None:
    from vercor.setups._data.veros_ocean import make_veros_data_ocean
    from vercor.setups._external.veros_data import snapshot_veros_state

    snapshot = snapshot_veros_state(native)
    ocean = make_veros_data_ocean(snapshot)
    assert ocean.grid is snapshot.grid
    assert ocean.spec.inputs == ()
    assert ocean.spec.transfer.time_selection == "linear"
    target = CallableComponent(
        "DST",
        snapshot.grid,
        lambda f: {},
        spec=ComponentSpec(inputs=("sea_surface_temperature",)),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 16, 5), 60.0, 1, calendar="noleap"),
        components=(ocean, target),
        exchanges=(
            Exchange(
                "OCN", "DST", ("sea_surface_temperature",), regridder_factory=bilinear
            ),
        ),
        run_order=("OCN", "DST"),
    )
    final = coupler.run()
    wet = np.asarray(snapshot.grid.binary_mask) > 0
    np.testing.assert_allclose(
        np.asarray(final.component("DST").field("sea_surface_temperature"))[wet], 273.15
    )


@pytest.mark.slow
def test_real_global_four_degree_snapshot_in_host_process() -> None:
    if importlib.util.find_spec("veros") is None:
        pytest.skip("optional Veros dependency is absent")
    code = """
import json
import numpy as np
from vercor.setups._external.veros_data import load_veros_data_snapshot
s = load_veros_data_snapshot()
assert s.grid.shape == (40, 90)
assert s.fields["sea_surface_temperature"].shape == (12, 40, 90)
assert (s.geometry["ht"] <= 0).all()
assert (s.geometry["area_t"] > 0).all()
t = s.geometry["maskT"]
# Native area/SSH stencils use the tracer and its west/south neighbour.
np.testing.assert_array_equal(s.geometry["maskU"], np.minimum(t, np.roll(t, 1, axis=0)))
south = np.concatenate((np.zeros_like(t[:, :1]), t[:, :-1]), axis=1)
np.testing.assert_array_equal(s.geometry["maskV"], np.minimum(t, south))
print(json.dumps({"shape": s.grid.shape, "wet": int(s.grid.binary_mask.sum())}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence["shape"] == [40, 90]
    assert evidence["wet"] > 0
