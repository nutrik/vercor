from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import vercor.setups._external._jax_gcm_pytree as jax_gcm_pytree_module
import vercor.setups._external.jax_gcm_tools as jax_gcm_tools_module
import vercor.setups._external.veros_fluxes as veros_fluxes_module
import vercor.setups._external.veros_state as veros_state_module
from tests.assertions import assert_allclose_compact
from vercor.dtypes import DTypePolicy
from vercor.fluxes import vertical_coordinates as vertical_coordinates_module
from vercor.physics import PhysicalConstants


class _FakeVariableStore:
    def __init__(self, **arrays: np.ndarray) -> None:
        for name, value in arrays.items():
            setattr(self, name, value)

    @contextmanager
    def unlock(self) -> Any:
        yield self


class _FakeVerosState:
    def __init__(self, variables: _FakeVariableStore) -> None:
        self.variables = variables
        self.settings = SimpleNamespace(__fields__=())


def test_change_and_get_default_jcm_parameter_values() -> None:
    parameters = jax_gcm_tools_module.Parameters.default()

    original = jax_gcm_tools_module.get_default_parameter_values(
        ["condensation.trlsc"],
        parameters,
    )
    assert np.isclose(
        float(original["condensation.trlsc"]), float(parameters.condensation.trlsc)
    )

    jax_gcm_tools_module.change_jcm_parameter_values(
        {"condensation.trlsc": 0.123},
        parameters,
    )

    updated = jax_gcm_tools_module.get_default_parameter_values(
        ["condensation.trlsc"],
        parameters,
    )
    assert np.isclose(float(updated["condensation.trlsc"]), 0.123)


def test_compute_sigma_pressure_levels_handles_valid_and_invalid_inputs() -> None:
    assert not hasattr(vertical_coordinates_module, "compute_pressure_levels")

    pressure = vertical_coordinates_module.compute_sigma_pressure_levels(
        reference_pressure=jnp.asarray(100000.0),
        top_pressure=jnp.asarray(10000.0),
        sigma_levels=jnp.asarray([0.0, 0.5, 1.0]),
        normalized_surface_pressure=jnp.asarray([[0.8, 1.0], [1.2, 0.6]]),
    )

    expected = np.asarray(
        [
            [[10000.0, 10000.0], [10000.0, 10000.0]],
            [[45000.0, 55000.0], [65000.0, 35000.0]],
            [[80000.0, 100000.0], [120000.0, 60000.0]],
        ]
    )
    assert_allclose_compact(pressure, expected)
    assert_allclose_compact(
        jax.jit(vertical_coordinates_module.compute_sigma_pressure_levels)(
            reference_pressure=jnp.asarray(100000.0),
            top_pressure=jnp.asarray(10000.0),
            sigma_levels=jnp.asarray([0.0, 0.5, 1.0]),
            normalized_surface_pressure=jnp.asarray([[0.8, 1.0], [1.2, 0.6]]),
        ),
        expected,
    )

    with pytest.raises(ValueError, match="top_pressure must be a scalar array"):
        vertical_coordinates_module.compute_sigma_pressure_levels(
            reference_pressure=jnp.asarray(100000.0),
            top_pressure=jnp.asarray([0.0, 1.0]),
            sigma_levels=jnp.asarray([0.0, 1.0]),
            normalized_surface_pressure=jnp.asarray([[1.0]]),
        )

    with pytest.raises(ValueError, match="sigma_levels must be a 1D array"):
        vertical_coordinates_module.compute_sigma_pressure_levels(
            reference_pressure=jnp.asarray(100000.0),
            top_pressure=jnp.asarray(0.0),
            sigma_levels=jnp.asarray([[0.0, 1.0]]),
            normalized_surface_pressure=jnp.asarray([[1.0]]),
        )


def test_compute_hybrid_pressure_levels_has_explicit_owner() -> None:
    surface_pressure = jnp.asarray([[100000.0, 90000.0], [80000.0, 70000.0]])
    hya = jnp.asarray([1000.0, 2000.0])
    hyb = jnp.asarray([0.25, 0.75])

    pressure = vertical_coordinates_module.compute_hybrid_pressure_levels(
        surface_pressure,
        hya,
        hyb,
    )

    expected = np.asarray(
        [
            [[26000.0, 77000.0], [23500.0, 69500.0]],
            [[21000.0, 62000.0], [18500.0, 54500.0]],
        ]
    )
    assert_allclose_compact(pressure, expected)


@pytest.mark.parametrize(
    ("input_data_directory", "expected_root"),
    [
        (None, Path("/tmp/jcm-resources")),
        (Path("/tmp/custom-jcm"), Path("/tmp/custom-jcm")),
    ],
)
def test_load_jcm_coords_terrain_forcing_uses_expected_paths(
    monkeypatch: pytest.MonkeyPatch,
    input_data_directory: Path | None,
    expected_root: Path,
) -> None:
    coords = SimpleNamespace(name="coords")
    calls: dict[str, Any] = {}

    def fake_get_speedy_coords(spectral_truncation: int) -> Any:
        calls["resolution"] = spectral_truncation
        return coords

    def fake_resource_files(package_name: str) -> Path:
        calls["package_name"] = package_name
        return expected_root

    def fake_terrain_from_file(path: Path, coords: Any) -> str:
        calls["terrain"] = (path, coords)
        return "terrain"

    def fake_forcing_from_file(path: Path, coords: Any) -> str:
        calls["forcing"] = (path, coords)
        return "forcing"

    monkeypatch.setattr(
        jax_gcm_tools_module,
        "get_speedy_coords",
        fake_get_speedy_coords,
    )
    monkeypatch.setattr(
        jax_gcm_tools_module.resources,
        "files",
        fake_resource_files,
    )
    monkeypatch.setattr(
        jax_gcm_tools_module.TerrainData,
        "from_file",
        staticmethod(fake_terrain_from_file),
    )
    monkeypatch.setattr(
        jax_gcm_tools_module.ForcingData,
        "from_file",
        staticmethod(fake_forcing_from_file),
    )

    actual_coords, terrain, forcing = (
        jax_gcm_tools_module.load_jcm_coords_terrain_forcing(
            resolution=21,
            input_data_directory=input_data_directory,
        )
    )

    assert actual_coords is coords
    assert terrain == "terrain"
    assert forcing == "forcing"
    assert calls["resolution"] == 21
    assert calls["terrain"] == (expected_root / "terrain.nc", coords)
    assert calls["forcing"] == (expected_root / "forcing.nc", coords)
    if input_data_directory is None:
        assert calls["package_name"] == "jcm.data.bc.t30.clim"


def test_load_jcm_inputs_scopes_h5netcdf_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = SimpleNamespace(name="coords")
    observed_orders: list[tuple[str, ...]] = []
    options = xr.get_options()
    if "netcdf_engine_order" not in options:
        pytest.skip("xarray does not support netcdf_engine_order")
    original_order = tuple(options["netcdf_engine_order"])

    monkeypatch.setattr(
        jax_gcm_tools_module,
        "get_speedy_coords",
        lambda spectral_truncation: coords,
    )

    def record_order(path: Path, coords: Any) -> str:
        _ = path, coords
        observed_orders.append(tuple(xr.get_options()["netcdf_engine_order"]))
        return "loaded"

    monkeypatch.setattr(
        jax_gcm_tools_module.TerrainData,
        "from_file",
        staticmethod(record_order),
    )
    monkeypatch.setattr(
        jax_gcm_tools_module.ForcingData,
        "from_file",
        staticmethod(record_order),
    )

    _, terrain, forcing = jax_gcm_tools_module.load_jcm_coords_terrain_forcing(
        input_data_directory=Path("/tmp/jcm-inputs"),
    )

    assert terrain == "loaded"
    assert forcing == "loaded"
    assert len(observed_orders) == 2
    assert all(order[0] == "h5netcdf" for order in observed_orders)
    assert tuple(xr.get_options()["netcdf_engine_order"]) == original_order


def test_prefer_h5netcdf_is_noop_without_engine_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jax_gcm_tools_module.xr, "get_options", lambda: {})

    def fail_if_called(**kwargs: Any) -> Any:
        pytest.fail(f"set_options must not be called: {kwargs}")

    monkeypatch.setattr(jax_gcm_tools_module.xr, "set_options", fail_if_called)

    with jax_gcm_tools_module._prefer_h5netcdf():
        pass


def test_load_jcm_inputs_restores_custom_engine_order_after_read_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = xr.get_options()
    if "netcdf_engine_order" not in options:
        pytest.skip("xarray does not support netcdf_engine_order")
    original_order = tuple(options["netcdf_engine_order"])
    custom_order = ("scipy", "h5netcdf", "netcdf4")
    observed_orders: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        jax_gcm_tools_module,
        "get_speedy_coords",
        lambda spectral_truncation: SimpleNamespace(name="coords"),
    )

    def raise_after_recording_order(path: Path, coords: Any) -> None:
        _ = path, coords
        observed_orders.append(tuple(xr.get_options()["netcdf_engine_order"]))
        raise RuntimeError("terrain read failed")

    monkeypatch.setattr(
        jax_gcm_tools_module.TerrainData,
        "from_file",
        staticmethod(raise_after_recording_order),
    )

    with xr.set_options(netcdf_engine_order=custom_order):
        with pytest.raises(RuntimeError, match="terrain read failed"):
            jax_gcm_tools_module.load_jcm_coords_terrain_forcing(
                input_data_directory=Path("/tmp/jcm-inputs"),
            )
        assert tuple(xr.get_options()["netcdf_engine_order"]) == custom_order

    assert observed_orders == [("h5netcdf", "scipy", "netcdf4")]
    assert tuple(xr.get_options()["netcdf_engine_order"]) == original_order


def test_jax_gcm_tree_helpers_transform_pytrees() -> None:
    tree = {
        "a": jnp.asarray([[1.0, 3.0], [5.0, 7.0]]),
        "b": jnp.asarray([[2.0, 4.0], [6.0, 8.0]]),
    }

    mean_tree = jax_gcm_pytree_module.tree_mean(tree, axis=0)
    assert_allclose_compact(mean_tree["a"], np.asarray([3.0, 5.0]))
    assert_allclose_compact(mean_tree["b"], np.asarray([4.0, 6.0]))

    unwrapped = jax_gcm_pytree_module.tree_unwrap_leading_dims(
        {
            "a": jnp.arange(24.0).reshape(2, 3, 4),
            "b": jnp.arange(24.0, 48.0).reshape(2, 3, 4),
        },
        first_n_dim=2,
    )
    assert unwrapped["a"].shape == (6, 4)
    assert unwrapped["b"].shape == (6, 4)

    stacked = jax_gcm_pytree_module.tree_stack(
        [
            {"a": jnp.asarray([1.0, 2.0]), "b": jnp.asarray([3.0, 4.0])},
            {"a": jnp.asarray([5.0, 6.0]), "b": jnp.asarray([7.0, 8.0])},
        ]
    )
    assert_allclose_compact(stacked["a"], np.asarray([[1.0, 2.0], [5.0, 6.0]]))
    assert_allclose_compact(stacked["b"], np.asarray([[3.0, 4.0], [7.0, 8.0]]))


def test_veros_compute_fluxes_preserves_sign_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (8, 8, 1, 1)
    variables = SimpleNamespace(
        u=np.arange(np.prod(shape), dtype=float).reshape(shape),
        v=2.0 * np.arange(np.prod(shape), dtype=float).reshape(shape),
        temp=np.full(shape, 5.0, dtype=float),
        maskT=np.ones((8, 8, 1), dtype=float),
        tau=0,
    )
    veros_state = SimpleNamespace(variables=variables)
    runtime_fields = {
        "model_level_height": np.full((4, 4), 10.0),
        "u_velocity": np.full((4, 4), 11.0),
        "v_velocity": np.full((4, 4), 12.0),
        "potential_temperature": np.full((4, 4), 13.0),
        "specific_humidity": np.full((4, 4), 14.0),
        "density": np.full((4, 4), 15.0),
        "temperature": np.full((4, 4), 16.0),
        "net_shortwave_radiation_flux": np.full((4, 4), 20.0),
        "downward_longwave_radiation_flux": np.full((4, 4), 30.0),
    }

    def fake_compute_ocean_surface_fluxes(*args: Any) -> tuple[Any, ...]:
        _ = args
        return (
            np.full((4, 4), 1.0),
            np.full((4, 4), 2.0),
            np.full((4, 4), 3.0),
            np.full((4, 4), 4.0),
            np.full((4, 4), 5.0),
            np.full((4, 4), 6.0),
            np.full((4, 4), 7.0),
            np.full((4, 4), 8.0),
            np.full((4, 4), 9.0),
            np.full((4, 4), 10.0),
            np.full((4, 4), 11.0),
            np.full((4, 4), 12.0),
            np.asarray(
                [
                    [-1e11, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ]
            ),
        )

    monkeypatch.setattr(
        veros_fluxes_module,
        "compute_ocean_surface_fluxes",
        fake_compute_ocean_surface_fluxes,
    )

    taux, tauy, qnet, qnec = veros_fluxes_module.compute_fluxes(
        veros_state=veros_state,  # type: ignore[arg-type]
        runtime_fields=runtime_fields,
        constants=PhysicalConstants(),
        dtype=DTypePolicy(),
    )

    assert_allclose_compact(taux, np.full((4, 4), 5.0))
    assert_allclose_compact(tauy, np.full((4, 4), 6.0))
    assert_allclose_compact(qnet, np.full((4, 4), 56.0))
    assert_allclose_compact(
        qnec,
        np.asarray(
            [
                [-0.0, -1.0, -2.0, -3.0],
                [-4.0, -5.0, -6.0, -7.0],
                [-8.0, -9.0, -10.0, -11.0],
                [-12.0, -13.0, -14.0, -15.0],
            ]
        ),
    )


def test_veros_state_helpers_cover_owned_host_mutation_paths() -> None:
    state = _FakeVerosState(
        variables=_FakeVariableStore(
            taux=np.zeros((8, 8, 1), dtype=float),
            tauy=np.zeros((8, 8, 1), dtype=float),
            qnet=np.zeros((8, 8, 1), dtype=float),
            qnec=np.zeros((8, 8, 1), dtype=float),
        )
    )

    calls: dict[str, int] = {"step": 0}

    def fake_step(current_state: Any) -> None:
        calls["step"] += 1
        current_state.marker = "updated"

    pure_state = veros_state_module.pure(
        state,
        step=fake_step,
        linear_solver=object(),
        execution="host",
    )
    assert pure_state is state
    assert calls["step"] == 1
    assert getattr(state, "marker") == "updated"

    forcing = veros_state_module.VerosForcingFields(
        taux=jnp.full((4, 4, 1), 9.0),
        tauy=jnp.full((4, 4, 1), 8.0),
        qnet=jnp.full((4, 4, 1), 7.0),
        qnec=jnp.full((4, 4, 1), 6.0),
    )
    result_state = veros_state_module.apply_veros_forcing_fields(
        state,
        forcing,
        execution="host",
    )

    assert result_state is state
    for name, expected in zip(
        ("taux", "tauy", "qnet", "qnec"),
        (9.0, 8.0, 7.0, 6.0),
        strict=True,
    ):
        value = getattr(state.variables, name)
        assert_allclose_compact(
            value[2:-2, 2:-2, :],
            np.full((4, 4, 1), expected),
        )
        assert_allclose_compact(value[:2, :, :], 0.0)
