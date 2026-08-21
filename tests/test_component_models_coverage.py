from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import numpy as np
from numpy.typing import NDArray
import pytest

import vercor.setups._data.era5_atmosphere as era5_atmosphere_module
import vercor.setups._data.era5_land as era5_land_module
import vercor.setups._data.era5_ocean as era5_ocean_module
import vercor.setups._data.erainterim_ocean as erainterim_ocean_module
import vercor.setups._data.jcm_land as jcm_land_module
from tests._coverage_support import CoverageCouplerStub, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.components import ComponentSpec, DataComponent
from vercor.components._adapter import normalize_component, prepare_component
from vercor.components.contexts import StepContext
from vercor.components.runtime_execution import step_component_runtime_state
from vercor.recipes import (
    ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_LAND_BASIC_FIELDS,
    ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
    ATMOSPHERE_TO_LAND_STATE_FIELDS,
    ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS,
    ATMOSPHERE_TO_OCEAN_STATE_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    LAND_TO_ATMOSPHERE_SOIL_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_SEAICE_SURFACE_FIELDS,
    SEAICE_TO_OCEAN_FIELDS,
    SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
    SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
)
from vercor.setups._data.era5_atmosphere import make_era5_atmosphere
from vercor.setups._data.era5_land import make_era5_land
from vercor.setups._data.era5_ocean import make_era5_ocean
from vercor.setups._data.erainterim_ocean import make_erainterim_ocean
from vercor.setups._data.jcm_land import make_jcm_land
from vercor.setups._slab.atmosphere import make_slab_atmosphere
from vercor.setups._slab.land import make_slab_land
from vercor.setups._slab.ocean import make_slab_ocean
from vercor.setups._slab.seaice import make_slab_seaice
from vercor._runtime.contracts import build_exchange_contracts
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.stores import FieldStore
from vercor.dtypes import DTypePolicy
from vercor._runtime.validation import validate_exchange_fields_declared
from vercor.exchanges import Exchange
from vercor.fields import _flatten_field_items
from vercor.regridding import bilinear, conservative
from vercor.types import RuntimeArray

_REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0
_DATA_ATMOSPHERE_INPUTS = tuple(
    _flatten_field_items(
        (*OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, *LAND_TO_ATMOSPHERE_SURFACE_FIELDS)
    )
)
_DATA_DRIVER_LAND_INPUTS = tuple(
    _flatten_field_items(
        (*ATMOSPHERE_TO_LAND_STATE_FIELDS, *ATMOSPHERE_TO_LAND_RADIATION_FIELDS)
    )
)
_DATA_LAND_INPUTS = tuple(
    dict.fromkeys(
        _flatten_field_items(
            (*_DATA_DRIVER_LAND_INPUTS, *ATMOSPHERE_TO_LAND_BASIC_FIELDS)
        )
    )
)
_DATA_OCEAN_INPUTS = tuple(
    _flatten_field_items(
        (*ATMOSPHERE_TO_OCEAN_STATE_FIELDS, *ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS)
    )
)
_DATA_OCEAN_PUBLIC_RECIPE_INPUTS = tuple(
    _flatten_field_items(ATMOSPHERE_TO_DATA_OCEAN_FIELDS)
)
_JCM_LAND_INPUTS = tuple(_flatten_field_items(ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS))


def _install_data_driver_factory_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch forcing readers with small arrays for data-driver contract tests."""

    atmosphere_forcing: dict[str, NDArray] = {
        "longitude": np.asarray([0.0, 180.0], dtype=float),
        "latitude": np.asarray([45.0, 0.0, -45.0], dtype=float),
        "hyai": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
        "hybi": np.asarray([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float),
        "hyam": np.asarray([20.0, 21.0, 22.0, 23.0], dtype=float),
        "hybm": np.asarray([30.0, 31.0, 32.0, 33.0], dtype=float),
        "lnsp": np.log(
            np.arange(1, 1 + (2 * 3 * 1 * 12), dtype=float).reshape(2, 3, 1, 12)
        ),
        "q": np.arange(1, 1 + (2 * 3 * 3 * 12), dtype=float).reshape(2, 3, 3, 12)
        / 1000.0,
        "t": 250.0 + np.arange(2 * 3 * 3 * 12, dtype=float).reshape(2, 3, 3, 12),
        "u": np.arange(1, 1 + (2 * 3 * 2 * 12), dtype=float).reshape(2, 3, 2, 12),
        "v": -np.arange(1, 1 + (2 * 3 * 2 * 12), dtype=float).reshape(2, 3, 2, 12),
        "msnswrf": np.full((2, 3, 12), 150.0, dtype=float),
        "msdwlwrf": np.full((2, 3, 12), 75.0, dtype=float),
    }
    land_forcing: dict[str, NDArray] = {
        "lon": np.asarray([0.0, 180.0], dtype=float),
        "lat": np.asarray([-45.0, 45.0], dtype=float),
        "mask": np.ones((2, 2), dtype=float),
        "skt": np.full((2, 2, 12), _REFERENCE_SURFACE_TEMPERATURE, dtype=float),
    }
    era5_ocean_forcing: dict[str, NDArray] = {
        "longitude": np.asarray([0.0, 180.0], dtype=float),
        "latitude": np.asarray([10.0, -10.0], dtype=float),
        "lsm": np.asarray(
            [
                [[0.0], [0.0]],
                [[0.0], [0.0]],
            ],
            dtype=float,
        ),
        "sst": np.full((2, 2, 12), _REFERENCE_SURFACE_TEMPERATURE, dtype=float),
    }
    yt = np.arange(-78.0, 82.0, 4.0)
    ocean_forcing: dict[str, NDArray] = {
        "xt": np.asarray([0.0, 4.0], dtype=float),
        "yt": yt,
        "sss": np.ones((2, yt.size, 12), dtype=float),
        "sst": np.full((2, yt.size, 12), 15.0, dtype=float),
    }

    def fake_atmosphere_data_path(file_type: str) -> Path:
        return Path(f"/tmp/{file_type}.nc")

    def fake_atmosphere_read(
        data_files: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        _ = data_files, where, flip_y
        return atmosphere_forcing[variable]

    def fake_land_data_path(file_type: str) -> Path:
        assert file_type == "era5_land_masked"
        return Path("/tmp/era5_land_masked.nc")

    def fake_land_read(
        data_files: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        _ = data_files, where, flip_y
        return land_forcing[variable]

    def fake_ocean_data_path(file_type: str) -> Path:
        assert file_type == "erainterim_ocean_4deg"
        return Path("/tmp/erainterim_ocean_4deg.nc")

    def fake_ocean_read(
        data_files: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        _ = data_files, where, flip_y
        return ocean_forcing[variable]

    def fake_era5_ocean_data_path(file_type: str) -> Path:
        assert file_type == "era5_surface"
        return Path("/tmp/era5_surface.nc")

    def fake_era5_ocean_read(
        data_files: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        _ = data_files, where, flip_y
        return era5_ocean_forcing[variable]

    monkeypatch.setattr(
        era5_atmosphere_module,
        "get_forcing_data",
        fake_atmosphere_data_path,
    )
    monkeypatch.setattr(
        era5_atmosphere_module,
        "_read_forcing",
        fake_atmosphere_read,
    )
    monkeypatch.setattr(era5_land_module, "get_forcing_data", fake_land_data_path)
    monkeypatch.setattr(era5_land_module, "_read_forcing", fake_land_read)
    monkeypatch.setattr(
        erainterim_ocean_module,
        "get_forcing_data",
        fake_ocean_data_path,
    )
    monkeypatch.setattr(erainterim_ocean_module, "_read_forcing", fake_ocean_read)
    monkeypatch.setattr(
        era5_ocean_module,
        "get_forcing_data",
        fake_era5_ocean_data_path,
    )
    monkeypatch.setattr(era5_ocean_module, "_read_forcing", fake_era5_ocean_read)


def _step_component(
    component: Any,
    dt: timedelta,
    time: datetime,
    coupler: Any,
) -> Any:
    """Advance one component through the runtime-state API."""

    _prepare_component_for_test(component, coupler)
    component_state = ComponentRuntimeState(
        fields=FieldStore.from_mapping(component._data),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
        payload=component._setup_payload,
    )
    return step_component_runtime_state(
        component,
        component_state,
        StepContext(
            dt_seconds=dt.total_seconds(),
            time=time,
            logger=coupler.logger,
        ),
        allow_host_runtime=False,
    )


def _prepare_component_for_test(component: Any, coupler: Any) -> None:
    """Run the public setup contract once and expose its frozen binding to tests."""

    if hasattr(component, "_data"):
        return
    binding = prepare_component(
        normalize_component(component),
        coupler.init_context(),
        DTypePolicy.from_jax_config(),
    )
    component._data = dict(binding._data)
    component._setup_payload = binding._payload


def _fake_jcm_land_inputs() -> tuple[Any, Any, Any]:
    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            longitudes=np.deg2rad(np.asarray([0.0, 180.0], dtype=float)),
            latitudes=np.deg2rad(np.asarray([-45.0, 45.0], dtype=float)),
        )
    )
    forcing = SimpleNamespace(
        stl_am=np.asarray([[280.0, 281.0], [282.0, 283.0]], dtype=float),
        soilw_am=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
    )
    return (
        coords,
        forcing,
        make_test_grid(
            name="ocn",
            longitude=np.asarray([0.0, 180.0], dtype=float),
            latitude=np.asarray([-45.0, 45.0], dtype=float),
            binary_mask=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        ),
    )


@pytest.mark.fast_always
def test_jcm_land_exchange_recipe_fields_are_declared() -> None:
    coords, forcing, ocn_grid = _fake_jcm_land_inputs()
    land = make_jcm_land(
        jcm_coords=cast(Any, coords),
        jcm_forcing=cast(Any, forcing),
        ocn_grid=ocn_grid,
    )
    atmosphere_grid = make_test_grid(name="atm")
    atmosphere = DataComponent(
        name="ATM",
        grid=atmosphere_grid,
        fields={
            field_name: np.zeros(atmosphere_grid.shape, dtype=float)
            for field_name in _JCM_LAND_INPUTS
        },
        spec=ComponentSpec(
            inputs=tuple(_flatten_field_items(JCM_LAND_TO_ATMOSPHERE_FIELDS)),
            outputs=_JCM_LAND_INPUTS,
            initial_fields={
                field_name: 0.0
                for field_name in _flatten_field_items(JCM_LAND_TO_ATMOSPHERE_FIELDS)
            },
        ),
    )
    components = {"ATM": atmosphere, "LND": land}
    contracts = build_exchange_contracts(
        tuple(components),
        (
            Exchange(
                source="ATM",
                target="LND",
                fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="LND",
                target="ATM",
                fields=JCM_LAND_TO_ATMOSPHERE_FIELDS,
                regridder_factory=bilinear,
            ),
        ),
        validate_endpoints=True,
    )

    assert contracts["LND"].receives == _JCM_LAND_INPUTS
    assert contracts["LND"].sends == tuple(
        _flatten_field_items(JCM_LAND_TO_ATMOSPHERE_FIELDS)
    )
    for name, component in components.items():
        validate_exchange_fields_declared(component, contracts[name])


@pytest.mark.fast_always
def test_data_driver_exchange_recipes_are_declared_by_data_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_data_driver_factory_fakes(monkeypatch)
    atmosphere = make_era5_atmosphere()
    ocean = make_erainterim_ocean()
    land = make_era5_land()
    components = {
        "ATM": atmosphere,
        "OCN": ocean,
        "LND": land,
    }
    exchanges = (
        Exchange(
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_OCEAN_STATE_FIELDS,
            route_id="atmosphere-ocean-state",
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS,
            route_id="atmosphere-ocean-radiation",
            regridder_factory=conservative,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=ATMOSPHERE_TO_LAND_STATE_FIELDS,
            route_id="atmosphere-land-state",
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
            route_id="atmosphere-land-radiation",
            regridder_factory=conservative,
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
            fields=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
    )
    contracts = build_exchange_contracts(
        tuple(components),
        exchanges,
        validate_endpoints=True,
    )

    assert contracts["ATM"].receives == _DATA_ATMOSPHERE_INPUTS
    assert contracts["OCN"].receives == _DATA_OCEAN_INPUTS
    assert contracts["LND"].receives == _DATA_DRIVER_LAND_INPUTS
    for name, component in components.items():
        validate_exchange_fields_declared(component, contracts[name])


@pytest.mark.fast_always
@pytest.mark.parametrize("make_ocean", (make_era5_ocean, make_erainterim_ocean))
def test_public_data_ocean_recipe_fields_are_declared_by_data_factories(
    monkeypatch: pytest.MonkeyPatch,
    make_ocean: Any,
) -> None:
    _install_data_driver_factory_fakes(monkeypatch)
    atmosphere_grid = make_test_grid(name="ATM")
    atmosphere = DataComponent(
        name="ATM",
        grid=atmosphere_grid,
        fields={
            field_name: np.zeros(atmosphere_grid.shape, dtype=float)
            for field_name in _DATA_OCEAN_PUBLIC_RECIPE_INPUTS
        },
        spec=ComponentSpec(outputs=_DATA_OCEAN_PUBLIC_RECIPE_INPUTS),
    )
    ocean = make_ocean()
    components = {"ATM": atmosphere, "OCN": ocean}
    contracts = build_exchange_contracts(
        tuple(components),
        (
            Exchange(
                source="ATM",
                target="OCN",
                fields=ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
                regridder_factory=bilinear,
            ),
        ),
        validate_endpoints=True,
    )

    assert contracts["OCN"].receives == _DATA_OCEAN_PUBLIC_RECIPE_INPUTS
    for name, component in components.items():
        validate_exchange_fields_declared(component, contracts[name])


@pytest.mark.fast_always
def test_slab_driver_exchange_recipes_are_declared_by_slab_factories() -> None:
    grid = make_test_grid(name="grid")
    components = {
        "ATM": make_slab_atmosphere(grid),
        "OCN": make_slab_ocean(grid),
        "LND": make_slab_land(grid),
        "ICE": make_slab_seaice(grid),
    }
    contracts = build_exchange_contracts(
        tuple(components),
        (
            Exchange(
                source="ATM",
                target="OCN",
                fields=SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="OCN",
                target="ATM",
                fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="OCN",
                target="ICE",
                fields=OCEAN_TO_SEAICE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="LND",
                target="ATM",
                fields=LAND_TO_ATMOSPHERE_SOIL_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
                regridder_factory=conservative,
            ),
            Exchange(
                source="ICE",
                target="OCN",
                fields=SEAICE_TO_OCEAN_FIELDS,
                regridder_factory=conservative,
            ),
        ),
        validate_endpoints=True,
    )

    assert contracts["ATM"].receives == (
        *_flatten_field_items(OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS),
        *_flatten_field_items(LAND_TO_ATMOSPHERE_SOIL_FIELDS),
    )
    assert contracts["OCN"].receives == (
        *_flatten_field_items(SLAB_ATMOSPHERE_TO_OCEAN_FIELDS),
        *_flatten_field_items(SEAICE_TO_OCEAN_FIELDS),
    )
    assert contracts["LND"].receives == tuple(
        _flatten_field_items(SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS)
    )
    assert contracts["ICE"].receives == tuple(
        _flatten_field_items(OCEAN_TO_SEAICE_SURFACE_FIELDS)
    )
    for name, component in components.items():
        validate_exchange_fields_declared(component, contracts[name])


@pytest.mark.fast_always
def test_slab_component_initialize_and_step_behaviors() -> None:
    coupler = cast(Any, CoverageCouplerStub())
    timestamp = datetime(2000, 1, 1, 0, 0, 0)
    dt = timedelta(hours=1)
    grid = make_test_grid(
        name="toy",
        longitude=np.asarray([0.0, 90.0]),
        latitude=np.asarray([-30.0, 30.0]),
    )

    atmosphere = cast(Any, make_slab_atmosphere(grid=grid))
    assert atmosphere.spec.inputs == (
        "sea_surface_temperature",
        "land_surface_temperature",
        "soil_moisture",
        "ice_fraction",
    )
    assert atmosphere.spec.outputs == (
        "temperature_2m",
        "sensible_heat_flux",
        "latent_heat_flux",
        "u_velocity_10m",
        "v_velocity_10m",
    )
    assert set(atmosphere.spec.initial_fields) == {
        "sea_surface_temperature",
        "land_surface_temperature",
        "soil_moisture",
        "ice_fraction",
        *atmosphere.spec.outputs,
    }
    atmosphere_state = _step_component(atmosphere, dt, timestamp, coupler)
    assert_allclose_compact(
        atmosphere_state.fields.get("sensible_heat_flux"),
        np.zeros(grid.shape),
    )
    assert_allclose_compact(
        atmosphere_state.fields.get("latent_heat_flux"),
        np.zeros(grid.shape),
    )

    atmosphere._data["sea_surface_temperature"] = np.asarray(
        [[280.0, 281.0], [282.0, 283.0]]
    )
    initial_temperature_2m = np.asarray(atmosphere._data["temperature_2m"]).copy()
    atmosphere_state = _step_component(atmosphere, dt, timestamp, coupler)
    atmosphere_data = atmosphere_state.fields
    assert atmosphere_data.get("sensible_heat_flux").shape == grid.shape
    assert atmosphere_data.get("latent_heat_flux").shape == grid.shape
    assert atmosphere_data.get("u_velocity_10m").shape == grid.shape
    assert atmosphere_data.get("v_velocity_10m").shape == grid.shape
    assert np.all(np.asarray(atmosphere_data.get("sensible_heat_flux")) < 0.0)
    assert np.all(np.asarray(atmosphere_data.get("latent_heat_flux")) > 0.0)
    assert np.any(np.asarray(atmosphere_data.get("u_velocity_10m")) != 0.0)
    assert np.any(np.asarray(atmosphere_data.get("v_velocity_10m")) != 0.0)
    assert np.all(
        np.asarray(atmosphere_data.get("temperature_2m")) < initial_temperature_2m
    )

    ocean = cast(Any, make_slab_ocean(grid=grid))
    assert ocean.spec.inputs == (
        "sensible_heat_flux",
        "latent_heat_flux",
        "u_velocity_10m",
        "v_velocity_10m",
        "u_velocity",
        "v_velocity",
        "specific_humidity",
        "temperature",
        "model_level_height",
        "net_shortwave_radiation_flux",
        "downward_longwave_radiation_flux",
        "ice_fraction",
    )
    assert ocean.spec.outputs == ("sea_surface_temperature",)
    assert set(ocean.spec.initial_fields) == {
        "sea_surface_temperature",
        "sensible_heat_flux",
        "latent_heat_flux",
        "u_velocity_10m",
        "v_velocity_10m",
        "u_velocity",
        "v_velocity",
        "specific_humidity",
        "temperature",
        "model_level_height",
        "net_shortwave_radiation_flux",
        "downward_longwave_radiation_flux",
        "ice_fraction",
    }
    ocean_state = _step_component(ocean, dt, timestamp, coupler)
    assert "sea_surface_temperature" in ocean_state.fields.field_names
    assert_allclose_compact(
        ocean_state.fields.get("sea_surface_temperature"),
        np.full(grid.shape, 288.15),
    )

    ocean._data["sensible_heat_flux"] = np.full(grid.shape, 20.0)
    ocean._data["latent_heat_flux"] = np.full(grid.shape, 10.0)
    starting_sst = ocean._data["sea_surface_temperature"].copy()
    ocean_state = _step_component(ocean, dt, timestamp, coupler)
    ocean_sst = ocean_state.fields.get("sea_surface_temperature")
    assert ocean_sst.shape == grid.shape
    assert np.all(np.asarray(ocean_sst) > starting_sst)

    land = cast(Any, make_slab_land(grid=grid))
    assert land.spec.inputs == ("latent_heat_flux", "sensible_heat_flux")
    assert land.spec.outputs == ("soil_moisture", "land_surface_temperature")
    assert set(land.spec.initial_fields) == {
        "soil_moisture",
        "land_surface_temperature",
        "latent_heat_flux",
        "sensible_heat_flux",
    }
    _prepare_component_for_test(land, coupler)
    land._data["latent_heat_flux"] = np.full(grid.shape, 100.0)
    land_state = _step_component(land, timedelta(seconds=10.0), timestamp, coupler)
    soil_moisture = land_state.fields.get("soil_moisture")
    assert soil_moisture.shape == grid.shape
    assert np.all(np.asarray(soil_moisture) < 0.3)

    seaice = cast(Any, make_slab_seaice(grid=grid))
    assert seaice.spec.inputs == ("sea_surface_temperature",)
    assert seaice.spec.outputs == ("ice_fraction",)
    assert set(seaice.spec.initial_fields) == {"ice_fraction"}
    seaice_state = _step_component(seaice, dt, timestamp, coupler)
    assert seaice_state.fields.field_names == ("ice_fraction",)
    assert_allclose_compact(
        seaice_state.fields.get("ice_fraction"), np.zeros(grid.shape)
    )

    _prepare_component_for_test(seaice, coupler)
    seaice._data["sea_surface_temperature"] = np.asarray(
        [[270.0, 272.0], [274.0, 276.0]]
    )
    seaice_state = _step_component(seaice, dt, timestamp, coupler)
    ice_fraction = seaice_state.fields.get("ice_fraction")
    cold = ice_fraction[0, 0]
    warm = ice_fraction[1, 1]
    assert cold > warm
    assert 0.0 < warm < 1.0


def test_era5_land_constructor_uses_masked_grid_and_enables_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_path = Path("/tmp/era5_land_masked.nc")
    forcing: dict[str, NDArray] = {
        "lon": np.asarray([0.0, 120.0, 240.0]),
        "lat": np.asarray([-30.0, 30.0]),
        "mask": np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "skt": np.asarray(
            [
                [[280.0], [281.0]],
                [[282.0], [283.0]],
                [[284.0], [285.0]],
            ]
        ),
    }

    def fake_get_forcing_data(file_type: str) -> Path:
        assert file_type == "era5_land_masked"
        return fake_path

    def fake_read_forcing(
        data_files: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        assert where == "surface"
        assert not flip_y
        return forcing[variable]

    monkeypatch.setattr(era5_land_module, "get_forcing_data", fake_get_forcing_data)
    monkeypatch.setattr(era5_land_module, "_read_forcing", fake_read_forcing)

    coupler = cast(Any, CoverageCouplerStub())
    component = cast(Any, make_era5_land())
    _step_component(component, timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert not hasattr(component, "_data_files")
    assert component.spec.transfer.time_selection == "linear"
    assert component.spec.inputs == _DATA_LAND_INPUTS
    assert component.spec.outputs == ("land_surface_temperature",)
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.grid.latitude, jax.Array)
    assert isinstance(component._data["land_surface_temperature"], jax.Array)
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
    assert isinstance(binary_mask, jax.Array)
    assert_allclose_compact(binary_mask, forcing["mask"].T)
    assert_allclose_compact(
        component._data["land_surface_temperature"],
        forcing["skt"].transpose((2, 1, 0)),
    )


def test_era5_ocean_constructor_applies_land_mask_and_reverses_latitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_path = Path("/tmp/era5_surface.nc")
    forcing: dict[str, NDArray] = {
        "longitude": np.asarray([0.0, 180.0]),
        "latitude": np.asarray([10.0, -10.0]),
        "lsm": np.asarray(
            [
                [[1.0], [0.4]],
                [[0.0], [1.0]],
            ]
        ),
        "sst": np.asarray(
            [
                [[280.0, 281.0], [282.0, 283.0]],
                [[284.0, 285.0], [286.0, 287.0]],
            ]
        ),
    }

    def fake_get_forcing_data(file_type: str) -> Path:
        assert file_type == "era5_surface"
        return fake_path

    def fake_read_forcing(
        data_files: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        assert where == "surface"
        return forcing[variable]

    monkeypatch.setattr(era5_ocean_module, "get_forcing_data", fake_get_forcing_data)
    monkeypatch.setattr(era5_ocean_module, "_read_forcing", fake_read_forcing)

    coupler = cast(Any, CoverageCouplerStub())
    component = cast(Any, make_era5_ocean())
    _step_component(component, timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert not hasattr(component, "_data_files")
    assert component.spec.transfer.time_selection == "linear"
    assert component.spec.inputs == _DATA_OCEAN_INPUTS
    assert component.spec.outputs == ("sea_surface_temperature",)
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component._data["sea_surface_temperature"], jax.Array)
    assert_allclose_compact(component.grid.latitude, np.asarray([-10.0, 10.0]))
    expected_mask = np.asarray([[0.0, 1.0], [0.0, 0.0]])
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
    assert_allclose_compact(binary_mask, expected_mask)
    assert component._data["sea_surface_temperature"].shape == (2, 2, 2)
    assert np.isnan(component._data["sea_surface_temperature"][0, 0, 0])
    assert np.isclose(component._data["sea_surface_temperature"][0, 0, 1], 284.0)


def test_erainterim_ocean_constructor_builds_global_masked_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_path = Path("/tmp/erainterim_4deg.nc")
    yt = np.arange(-78.0, 82.0, 4.0)
    sss = np.ones((2, yt.size, 12), dtype=float)
    sst = 10.0 * np.ones((2, yt.size, 12), dtype=float)
    forcing: dict[str, NDArray] = {
        "xt": np.asarray([0.0, 4.0]),
        "yt": yt,
        "sss": sss,
        "sst": sst,
    }

    def fake_get_forcing_data(file_type: str) -> Path:
        assert file_type == "erainterim_ocean_4deg"
        return fake_path

    def fake_read_forcing(
        data_files: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        assert where == "model_level"
        assert not flip_y
        return forcing[variable]

    monkeypatch.setattr(
        erainterim_ocean_module,
        "get_forcing_data",
        fake_get_forcing_data,
    )
    monkeypatch.setattr(erainterim_ocean_module, "_read_forcing", fake_read_forcing)

    coupler = cast(Any, CoverageCouplerStub())
    component = cast(Any, make_erainterim_ocean(resolution="4deg"))
    _step_component(component, timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert not hasattr(component, "_data_files")
    assert component.spec.transfer.time_selection == "linear"
    assert component.spec.inputs == _DATA_OCEAN_INPUTS
    assert component.spec.outputs == ("sea_surface_temperature",)
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component._data["sea_surface_temperature"], jax.Array)
    assert component.grid.shape == (46, 2)
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
    assert np.all(binary_mask[3:-3, :] == 1.0)
    assert np.all(binary_mask[:3, :] == 0.0)
    assert component._data["sea_surface_temperature"].shape == (12, 46, 2)
    assert np.isnan(component._data["sea_surface_temperature"][0, 0, 0])
    assert np.isclose(component._data["sea_surface_temperature"][0, 3, 0], 283.15)


def test_era5_atmosphere_constructor_initialize_and_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_level_path = Path("/tmp/era5_model_levels.nc")
    surface_path = Path("/tmp/era5_surface.nc")

    forcing: dict[str, NDArray] = {
        "longitude": np.asarray([0.0, 180.0], dtype=float),
        "latitude": np.asarray([45.0, 0.0, -45.0], dtype=float),
        "hyai": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
        "hybi": np.asarray([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float),
        "hyam": np.asarray([20.0, 21.0, 22.0, 23.0], dtype=float),
        "hybm": np.asarray([30.0, 31.0, 32.0, 33.0], dtype=float),
        "lnsp": np.log(
            np.arange(1, 1 + (2 * 3 * 1 * 12), dtype=float).reshape(2, 3, 1, 12)
        ),
        "q": np.arange(1, 1 + (2 * 3 * 3 * 12), dtype=float).reshape(2, 3, 3, 12)
        / 1000.0,
        "t": 250.0 + np.arange(2 * 3 * 3 * 12, dtype=float).reshape(2, 3, 3, 12),
        "u": np.arange(1, 1 + (2 * 3 * 2 * 12), dtype=float).reshape(2, 3, 2, 12),
        "v": -np.arange(1, 1 + (2 * 3 * 2 * 12), dtype=float).reshape(2, 3, 2, 12),
        "msnswrf": np.full((2, 3, 12), 150.0, dtype=float),
        "msdwlwrf": np.full((2, 3, 12), 75.0, dtype=float),
    }

    physics_calls: dict[str, list[Any]] = {
        "pressure": [],
        "height": [],
        "density": [],
        "theta": [],
    }

    def fake_get_forcing_data(file_type: str) -> Path:
        if file_type == "era5_model_levels":
            return model_level_path
        if file_type == "era5_surface":
            return surface_path
        raise AssertionError(f"Unexpected forcing lookup: {file_type}")

    def fake_read_forcing(
        data_files: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        if where == "model_level":
            assert variable in {
                "longitude",
                "latitude",
                "hyai",
                "hybi",
                "hyam",
                "hybm",
                "lnsp",
                "q",
                "t",
                "u",
                "v",
            }
        else:
            assert where == "surface"
            assert variable in {"msnswrf", "msdwlwrf"}
        _ = flip_y
        return forcing[variable]

    def fake_compute_hybrid_pressure_levels(
        surface_pressure: NDArray,
        hya: NDArray,
        hyb: NDArray,
    ) -> NDArray:
        physics_calls["pressure"].append(
            (surface_pressure.copy(), hya.copy(), hyb.copy())
        )
        base = float(surface_pressure.mean())
        nlev = hya.size
        return np.stack(
            [
                np.full(surface_pressure.shape, base + float(level), dtype=float)
                for level in range(nlev)
            ],
            axis=2,
        )

    def fake_get_altitudes_hybrid_sigma_levels(
        settings: Any,
        temperature_3d: NDArray,
        specific_humidity_3d: NDArray,
        pressure_h: NDArray,
    ) -> NDArray:
        physics_calls["height"].append(
            (
                settings,
                temperature_3d.copy(),
                specific_humidity_3d.copy(),
                pressure_h.copy(),
            )
        )
        height = np.zeros((3, 2, 2), dtype=float)
        height[..., 1] = float(pressure_h.mean())
        return height

    def fake_compute_air_density(
        settings: Any,
        pressure_level: NDArray,
        temperature: NDArray,
    ) -> NDArray:
        physics_calls["density"].append(
            (settings, pressure_level.copy(), temperature.copy())
        )
        return np.asarray(pressure_level + temperature, dtype=float)

    def fake_compute_potential_temperature(
        settings: Any,
        temperature: NDArray,
        pressure_level: NDArray,
    ) -> NDArray:
        physics_calls["theta"].append(
            (settings, temperature.copy(), pressure_level.copy())
        )
        return np.asarray(pressure_level - temperature, dtype=float)

    monkeypatch.setattr(
        era5_atmosphere_module,
        "get_forcing_data",
        fake_get_forcing_data,
    )
    monkeypatch.setattr(era5_atmosphere_module, "_read_forcing", fake_read_forcing)
    monkeypatch.setattr(
        era5_atmosphere_module,
        "compute_hybrid_pressure_levels",
        fake_compute_hybrid_pressure_levels,
    )
    monkeypatch.setattr(
        era5_atmosphere_module,
        "get_altitudes_hybrid_sigma_levels",
        fake_get_altitudes_hybrid_sigma_levels,
    )
    monkeypatch.setattr(
        era5_atmosphere_module,
        "compute_air_density",
        fake_compute_air_density,
    )
    monkeypatch.setattr(
        era5_atmosphere_module,
        "compute_potential_temperature",
        fake_compute_potential_temperature,
    )

    coupler = cast(Any, CoverageCouplerStub())
    component = cast(Any, make_era5_atmosphere())
    _prepare_component_for_test(component, coupler)

    assert not hasattr(component, "_data_files")
    assert not hasattr(component, "_hybrid_coefficients")
    assert component.spec.transfer.time_selection == "linear"
    assert component.spec.inputs == _DATA_ATMOSPHERE_INPUTS
    assert set(_DATA_ATMOSPHERE_INPUTS).issubset(component.spec.initial_fields)
    assert {
        "surface_pressure",
        "specific_humidity_3d",
        "temperature_3d",
        "u_velocity",
        "v_velocity",
    }.issubset(component.spec.initial_fields)
    assert {
        "model_level_height",
        "density",
        "potential_temperature",
    }.isdisjoint(component.spec.initial_fields)
    assert component.spec.outputs == (
        "surface_pressure",
        "specific_humidity_3d",
        "temperature_3d",
        "u_velocity",
        "v_velocity",
        "net_shortwave_radiation_flux",
        "downward_longwave_radiation_flux",
        "specific_humidity",
        "temperature",
        "model_level_height",
        "density",
        "potential_temperature",
    )
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component._data["surface_pressure"], jax.Array)
    assert_allclose_compact(component.grid.longitude, forcing["longitude"])
    assert_allclose_compact(component.grid.latitude, np.asarray([-45.0, 0.0, 45.0]))
    for coefficient_name in ("hyai", "hybi", "hyam", "hybm"):
        assert coefficient_name not in component._data
    assert component._data["surface_pressure"].shape == (12, 3, 2)
    assert component._data["specific_humidity_3d"].shape == (12, 2, 3, 2)
    assert component._data["temperature_3d"].shape == (12, 2, 3, 2)
    assert component._data["u_velocity"].shape == (12, 3, 2)
    assert component._data["v_velocity"].shape == (12, 3, 2)

    assert component.spec.outputs == (
        "surface_pressure",
        "specific_humidity_3d",
        "temperature_3d",
        "u_velocity",
        "v_velocity",
        "net_shortwave_radiation_flux",
        "downward_longwave_radiation_flux",
        "specific_humidity",
        "temperature",
        "model_level_height",
        "density",
        "potential_temperature",
    )
    assert len(physics_calls["pressure"]) == 24
    assert len(physics_calls["height"]) == 12
    assert len(physics_calls["density"]) == 12
    assert len(physics_calls["theta"]) == 12
    assert_allclose_compact(
        physics_calls["pressure"][0][1], np.asarray([2.0, 3.0, 4.0])
    )
    assert_allclose_compact(
        physics_calls["pressure"][0][2], np.asarray([12.0, 13.0, 14.0])
    )
    assert component._data["model_level_height"].shape == (12, 3, 2)
    assert component._data["density"].shape == (12, 3, 2)
    assert component._data["potential_temperature"].shape == (12, 3, 2)
    assert np.all(component._data["model_level_height"] > 0.0)
    assert "total_surface_temperature" not in component._data


def test_jcm_land_constructor_converts_coords_and_preserves_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    recorded_inputs: dict[str, NDArray] = {}

    def fake_create_lnd_mask_from_ocn(
        atm_lat: NDArray,
        atm_lon: NDArray,
        ocn_grid: Any,
    ) -> tuple[NDArray, NDArray]:
        recorded_inputs["atm_lat"] = np.asarray(atm_lat)
        recorded_inputs["atm_lon"] = np.asarray(atm_lon)
        recorded_inputs["ocn_lon"] = np.asarray(ocn_grid.longitude)
        return expected_mask, np.zeros_like(expected_mask)

    monkeypatch.setattr(
        jcm_land_module,
        "create_lnd_mask_from_ocn",
        fake_create_lnd_mask_from_ocn,
    )

    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            longitudes=np.deg2rad(np.asarray([0.0, 180.0], dtype=float)),
            latitudes=np.deg2rad(np.asarray([-45.0, 45.0], dtype=float)),
        )
    )
    forcing = SimpleNamespace(
        stl_am=np.asarray([[280.0, 281.0], [282.0, 283.0]], dtype=float),
        soilw_am=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
    )
    ocn_grid = make_test_grid(name="ocn")

    component = cast(
        Any,
        make_jcm_land(
            jcm_coords=cast(Any, coords),
            jcm_forcing=cast(Any, forcing),
            ocn_grid=ocn_grid,
        ),
    )
    coupler = cast(Any, CoverageCouplerStub())
    _step_component(component, timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert component.spec.transfer.time_selection == "daily"
    assert component.spec.outputs == (
        "land_surface_temperature",
        "soil_moisture",
    )
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component._data["land_surface_temperature"], jax.Array)
    assert isinstance(component._data["soil_moisture"], jax.Array)
    assert_allclose_compact(recorded_inputs["atm_lon"], np.asarray([0.0, 180.0]))
    assert_allclose_compact(recorded_inputs["atm_lat"], np.asarray([-45.0, 45.0]))
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
    assert_allclose_compact(binary_mask, expected_mask)
    assert_allclose_compact(
        component._data["land_surface_temperature"],
        forcing.stl_am.T,
    )
    assert_allclose_compact(component._data["soil_moisture"], forcing.soilw_am.T)


def test_jcm_land_canonicalizes_jcm_2_time_series_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        jcm_land_module,
        "create_lnd_mask_from_ocn",
        lambda **kwargs: (np.ones((2, 2)), np.zeros((2, 2))),
    )
    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            longitudes=np.deg2rad(np.asarray([0.0, 180.0])),
            latitudes=np.deg2rad(np.asarray([-45.0, 45.0])),
        )
    )
    values = np.arange(12.0).reshape(3, 2, 2)
    forcing = SimpleNamespace(
        stl_am=SimpleNamespace(values=values),
        soilw_am=SimpleNamespace(values=values + 20.0),
    )

    component = make_jcm_land(
        cast(Any, coords),
        cast(Any, forcing),
        make_test_grid(name="ocn"),
    )
    _prepare_component_for_test(component, cast(Any, CoverageCouplerStub()))

    assert_allclose_compact(
        cast(Any, component)._data["land_surface_temperature"],
        values.transpose(0, 2, 1),
    )


def test_jcm_land_rejects_invalid_jcm_forcing_rank() -> None:
    with pytest.raises(
        ValueError,
        match="JCM forcing field 'stl_am'.*shape \\(3,\\)",
    ):
        jcm_land_module._canonicalize_jcm_forcing_field(
            np.ones(3),
            field_name="stl_am",
        )
