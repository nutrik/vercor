from __future__ import annotations

from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, cast, get_type_hints
import ast
import importlib
import subprocess
import sys

import jax.numpy as jnp
import pytest

import vercor
import vercor.exceptions as exceptions_module
import vercor.components as components_module
import vercor.components.base as base_module
import vercor.components.contexts as component_contexts_module
import vercor.components.contracts as component_contracts_module
import vercor.components.data as data_module
import vercor.components.setup_validation as setup_validation_module
from tests._architecture_support import package_import_cycles
from tests._coverage_support import make_test_grid
from vercor.components import (
    CallableComponent,
    Component,
    ComponentSpec,
    DataComponent,
    StepResult,
)
from vercor.calendar import DateTime360
from vercor.clock import Clock
from vercor.coupler import Coupler
from vercor.exceptions import (
    AssetError,
    ComponentError,
    CouplerError,
    ExchangeError,
    GridError,
    RegridderError,
)
from vercor.exchanges import Exchange
from vercor.fields import (
    COMMON_FIELD_NAMES,
    VectorField,
    _flatten_field_items,
    vector,
)
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.stores import FieldStore
from vercor.output import OutputTarget
from vercor.regridding import bilinear
from vercor.state import ComponentState, RunState


@pytest.mark.fast_always
def test_public_api_exports_state_view_fields_and_regridders() -> None:
    from vercor import RectilinearGrid
    from vercor.regridding import (
        Regridder,
        RegridderFactory,
        bilinear as public_bilinear,
        conservative as public_conservative,
    )

    assert not hasattr(vercor, "VectorField")
    assert not hasattr(vercor, "vector")
    assert not hasattr(vercor, "CouplerSpec")
    assert RectilinearGrid is vercor.grids.RectilinearGrid
    assert public_bilinear is bilinear
    assert callable(public_conservative)
    assert RunState.__name__ == "RunState"
    assert ComponentState.__name__ == "ComponentState"
    assert getattr(Regridder, "_is_protocol", False)
    assert cast(Any, RegridderFactory).__name__ == "RegridderFactory"
    assert {"RunState", "RectilinearGrid"}.issubset(vercor.__all__)
    assert {"ComponentState", "VectorField", "vector"}.isdisjoint(vercor.__all__)
    assert "Regridder" not in vercor.__all__
    assert "RegridderFactory" not in vercor.__all__
    assert "grid_from_coordinates" not in vercor.__all__
    assert "uniform_rectilinear_grid" not in vercor.__all__


@pytest.mark.fast_always
def test_root_api_is_core_only_after_boundary_redesign() -> None:
    allowed_root_exports = {
        "Clock",
        "Coupler",
        "Exchange",
        "RectilinearGrid",
        "RuntimeOptions",
        "RunState",
    }

    assert set(vercor.__all__) == allowed_root_exports
    for name in (
        "bilinear",
        "conservative",
        "Regridder",
        "RegridderFactory",
        "CAMulatorConfig",
        "fluxes",
        "JAXGCMConfig",
        "recipes",
        "setups",
        "Spinup",
        "SurfaceMaskPolicy",
        "VerosConfig",
    ):
        assert name not in vercor.__all__
    for name in ("bilinear", "conservative", "Regridder", "RegridderFactory"):
        assert not hasattr(vercor, name)


@pytest.mark.fast_always
def test_setup_implementation_modules_are_private_after_boundary_redesign() -> None:
    import vercor.setups as setup_facade

    public_factory_names = {
        "CAMulatorConfig",
        "JAXGCMConfig",
        "JCMLandAtmosphereConfig",
        "JCMLandAtmosphereSetup",
        "load_jcm_inputs",
        "make_camulator_gcm",
        "make_camulator_land",
        "make_era5_atmosphere",
        "make_era5_land",
        "make_era5_ocean",
        "make_erainterim_ocean",
        "make_jax_gcm",
        "make_jcm_land",
        "make_jcm_land_atmosphere",
        "make_slab_atmosphere",
        "make_slab_land",
        "make_slab_ocean",
        "make_slab_seaice",
        "make_veros_gcm",
        "Spinup",
        "VerosConfig",
    }

    assert public_factory_names.issubset(set(setup_facade.__all__))
    with pytest.raises(ModuleNotFoundError, match="vercor.setup_config"):
        importlib.import_module("vercor.setup_config")
    for module_name in (
        "vercor.setups.data",
        "vercor.setups.data.era5_land",
        "vercor.setups.external",
        "vercor.setups.external.jax_gcm_state",
        "vercor.setups.slab",
        "vercor.setups.jcm_setup_helpers",
    ):
        missing_root = ".".join(module_name.split(".")[:3])
        with pytest.raises(ModuleNotFoundError, match=missing_root):
            importlib.import_module(module_name)


@pytest.mark.fast_always
def test_boundary_redesign_removes_remaining_duplicate_public_helpers() -> None:
    import vercor.fields as fields_module
    import vercor.grids as grids_module
    from vercor.state import ComponentState

    with pytest.raises(ModuleNotFoundError, match="vercor.config"):
        importlib.import_module("vercor.config")
    assert hasattr(fields_module, "COMMON_FIELD_NAMES")
    assert "sea_surface_temperature" in fields_module.COMMON_FIELD_NAMES
    assert not hasattr(fields_module, "VALID_FIELD_NAMES")
    assert fields_module.__all__ == [
        "COMMON_FIELD_NAMES",
        "ExchangeField",
        "VectorField",
        "vector",
    ]
    assert not hasattr(grids_module, "Grid")
    assert not hasattr(ComponentState, "field_candidates")


@pytest.mark.fast_always
def test_public_api_uses_current_owners_and_facades() -> None:
    import vercor.exchanges as exchanges_module
    import vercor.fields as fields_module
    import vercor.grids as grids_module
    import vercor.recipes as recipes_module

    assert vercor.RectilinearGrid.__module__ == "vercor.grids"
    assert VectorField.__module__ == "vercor.fields"
    assert vercor.Exchange.__module__ == "vercor.exchanges"
    assert vercor.RunState.__module__ == "vercor.state"
    assert ComponentState.__module__ == "vercor.state"
    assert vercor.RectilinearGrid is grids_module.RectilinearGrid
    assert "grid_from_coordinates" not in vercor.__all__

    grid = grids_module.RectilinearGrid.from_coordinates(
        "explicit-grid",
        longitude=jnp.asarray([0.0, 90.0]),
        latitude=jnp.asarray([-45.0, 45.0]),
    )
    assert isinstance(grid, grids_module.RectilinearGrid)
    assert grid.shape == (2, 2)

    assert fields_module.__all__ == [
        "COMMON_FIELD_NAMES",
        "ExchangeField",
        "VectorField",
        "vector",
    ]
    assert recipes_module.OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS == (
        "sea_surface_temperature",
    )
    assert "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS" in recipes_module.__all__
    assert "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS" not in exchanges_module.__all__
    state_source = Path("vercor/state.py").read_text(encoding="utf-8")
    assert "__module__" not in state_source


@pytest.mark.fast_always
def test_public_facades_hide_private_implementation_modules() -> None:
    import vercor.output as output_module
    import vercor.regridding as regridding_module
    import vercor.state as state_module
    from vercor.output import (
        OutputSpec,
        OutputVariable,
        PeriodOutput,
    )

    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")
    exchange_signature = str(signature(Exchange))
    exchange_parameters = signature(Exchange).parameters
    public_bilinear_signature = str(signature(regridding_module.bilinear))
    public_conservative_signature = str(signature(regridding_module.conservative))

    assert "from vercor.exchanges import Exchange" in coupler_source
    assert Exchange.__module__ == "vercor.exchanges"
    assert "_exchange" not in exchange_signature
    assert "_regridders" not in exchange_signature
    assert "RegridderFactory" in exchange_signature
    assert exchange_parameters["route_id"].kind is Parameter.KEYWORD_ONLY
    assert exchange_parameters["regridder_factory"].kind is Parameter.KEYWORD_ONLY
    assert regridding_module.bilinear.__module__ == "vercor.regridding"
    assert regridding_module.conservative.__module__ == "vercor.regridding"
    assert not hasattr(vercor, "bilinear")
    assert not hasattr(vercor, "conservative")
    assert "_regridders" not in public_bilinear_signature
    assert "_regridders" not in public_conservative_signature
    assert "Regridder" in public_bilinear_signature
    assert "Regridder" in public_conservative_signature
    assert regridding_module.__all__ == [
        "Regridder",
        "RegridderFactory",
        "VectorRegridder",
        "bilinear",
        "conservative",
    ]
    assert state_module.__all__ == [
        "ComponentState",
        "FieldLookupScope",
        "FieldScope",
        "RunState",
    ]
    assert output_module.__all__ == [
        "OutputContext",
        "OutputFrame",
        "OutputProvider",
        "OutputSpec",
        "OutputTarget",
        "OutputVariable",
        "PeriodOutput",
        "SnapshotContext",
        "SnapshotWriter",
    ]
    assert OutputSpec is output_module.OutputSpec
    assert OutputVariable is output_module.OutputVariable
    assert PeriodOutput is output_module.PeriodOutput
    assert OutputSpec.__module__ == "vercor.output"
    assert PeriodOutput.__module__ == "vercor.output"


@pytest.mark.fast_always
def test_fields_facade_owns_vector_field_contract() -> None:
    field = vector("u_velocity", "v_velocity")

    assert field == VectorField("u_velocity", "v_velocity")
    assert _flatten_field_items(("temperature", field)) == [
        "temperature",
        "u_velocity",
        "v_velocity",
    ]

    with pytest.raises(
        TypeError, match="Tuple vector field declarations are unsupported"
    ):
        Exchange("ATM", "OCN", (("u_velocity", "v_velocity"),))  # type: ignore[arg-type]


@pytest.mark.fast_always
def test_state_constructors_do_not_expose_runtime_stores() -> None:
    run_state_signature = str(signature(vercor.RunState))
    component_state_signature = signature(ComponentState)

    assert "ComponentRuntimeState" not in run_state_signature
    assert "FieldStore" not in run_state_signature
    assert "components" not in component_state_signature.parameters
    assert "data" not in component_state_signature.parameters
    assert "FieldStore" not in str(component_state_signature)
    assert "fields" in component_state_signature.parameters

    with pytest.raises(TypeError, match="Coupler"):
        vercor.RunState()


@pytest.mark.fast_always
def test_private_regridder_base_does_not_shadow_public_protocol() -> None:
    base_module = importlib.import_module("vercor._regridders.base")

    assert hasattr(base_module, "_BaseRegridder")
    assert not hasattr(base_module, "Regridder")
    assert "class _BaseRegridder" in Path("vercor/_regridders/base.py").read_text(
        encoding="utf-8"
    )


@pytest.mark.fast_always
def test_runtime_internals_live_under_private_runtime_package() -> None:
    with pytest.raises(ModuleNotFoundError, match="vercor.runtime.state"):
        importlib.import_module("vercor.runtime.state")

    runtime_state = importlib.import_module("vercor._runtime.state")
    runtime_stores = importlib.import_module("vercor._runtime.stores")
    runtime_contracts = importlib.import_module("vercor._runtime.contracts")
    runtime_facade = importlib.import_module("vercor._runtime.facade")

    assert hasattr(runtime_state, "ComponentRuntimeState")
    assert not hasattr(runtime_state, "RuntimeComponentState")
    assert hasattr(runtime_stores, "FieldStore")
    assert not hasattr(runtime_stores, "RuntimeFieldStore")
    assert hasattr(runtime_contracts, "ExchangeContract")
    assert not hasattr(runtime_contracts, "RuntimeComponentContract")
    runtime_prepared = importlib.import_module("vercor._runtime.prepared")
    assert hasattr(runtime_prepared, "PreparedCoupling")
    assert not hasattr(runtime_facade, "RuntimeInputs")


@pytest.mark.fast_always
def test_runtime_private_state_uses_public_domain_vocabulary() -> None:
    state_parameters = signature(ComponentRuntimeState).parameters
    contract_parameters = signature(
        importlib.import_module("vercor._runtime.contracts").ExchangeContract
    ).parameters
    component_state_parameters = signature(ComponentState).parameters

    assert tuple(state_parameters) == ("fields", "received", "sent", "payload")
    assert tuple(contract_parameters) == ("receives", "sends")
    assert "received" in component_state_parameters
    assert "sent" in component_state_parameters
    assert "incoming" not in component_state_parameters
    assert "outgoing" not in component_state_parameters
    assert not hasattr(vercor.state, "runtime_field")
    assert not hasattr(vercor.state, "runtime_field_candidates")


@pytest.mark.fast_always
def test_field_name_deduplication_has_one_private_owner() -> None:
    assert not Path("vercor/components/_field_names.py").exists()
    assert Path("vercor/_field_names.py").exists()
    assert "def unique_field_names(" not in Path("vercor/fields.py").read_text(
        encoding="utf-8"
    )


@pytest.mark.fast_always
def test_coupler_public_methods_return_stable_state_and_views(
    tmp_path: Path,
) -> None:
    component = DataComponent(
        name="ATM",
        grid=make_test_grid(name="coupler-validated"),
        fields={"temperature": 280.0},
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    assert not hasattr(coupler, "add_exchanges")
    assert not hasattr(Coupler, "initialize")
    assert not hasattr(Coupler, "state")
    assert not hasattr(Coupler, "view")
    assert not hasattr(Coupler, "views")
    assert get_type_hints(Coupler.initial_state)["return"] is RunState
    assert get_type_hints(Coupler.run)["return"] is RunState

    state = coupler.initial_state()
    assert isinstance(state, RunState)
    view = state.component("ATM")
    assert isinstance(view, ComponentState)

    output_state = coupler.run(
        state,
        output=OutputTarget(
            tmp_path,
            write_period=False,
            write_final_fields=False,
            write_snapshots=False,
        ),
    )
    assert isinstance(output_state, RunState)
    assert not hasattr(Coupler, "write_outputs")
    with pytest.raises(TypeError, match="snapshots"):
        coupler.run(state, snapshots=False)  # type: ignore[call-arg]


@pytest.mark.fast_always
def test_data_component_and_grid_constructors_use_keyword_vocabulary() -> None:
    grid = vercor.RectilinearGrid.uniform(
        "custom-grid",
        nlon=2,
        nlat=2,
        longitude=(0.0, 90.0),
        latitude=(-45.0, 45.0),
        binary_mask=jnp.ones((2, 2)),
    )

    component = DataComponent(
        "ATM",
        grid,
        fields={"temperature": 280.0},
    )

    assert grid.binary_mask is not None
    assert component.spec.outputs == ("temperature",)
    with pytest.raises(TypeError, match="lon"):
        vercor.RectilinearGrid.uniform(
            "old-grid",
            nlon=2,
            nlat=2,
            lon=(0.0, 90.0),  # type: ignore[call-arg]
            lat=(-45.0, 45.0),  # type: ignore[call-arg]
        )
    positional = DataComponent("ATM", grid, {"humidity": 0.5})
    assert positional.spec.outputs == ("humidity",)


@pytest.mark.fast_always
def test_component_constructor_hides_raw_setup_internals() -> None:
    for component_type in (Component, CallableComponent, DataComponent):
        parameters = signature(component_type).parameters
        assert "data" not in parameters
        assert "setup_metadata" not in parameters
        assert "payload" not in parameters


@pytest.mark.fast_always
def test_regridders_expose_explicit_scalar_and_vector_methods() -> None:
    from vercor.regridding import Regridder, VectorRegridder

    grid = vercor.RectilinearGrid.from_coordinates(
        "regrid-methods",
        longitude=jnp.asarray([0.0, 90.0]),
        latitude=jnp.asarray([-45.0, 45.0]),
    )
    regridder = bilinear(grid, grid)
    scalar = jnp.ones(grid.shape)
    u = jnp.ones(grid.shape)
    v = -jnp.ones(grid.shape)

    assert isinstance(regridder, Regridder)
    assert isinstance(regridder, VectorRegridder)
    assert regridder.target_grid is grid
    assert regridder.regrid(scalar) is scalar
    assert regridder.regrid_vector(u, v) == (u, v)


@pytest.mark.fast_always
def test_top_level_exports_public_exceptions() -> None:
    assert issubclass(ComponentError, CouplerError)
    assert issubclass(GridError, CouplerError)
    assert issubclass(RegridderError, CouplerError)
    assert issubclass(ExchangeError, CouplerError)
    assert issubclass(AssetError, Exception)
    assert "ExchangeError" not in vercor.__all__
    assert not hasattr(vercor, "ExchangeError")
    assert not hasattr(vercor, "ExchangerError")
    assert exceptions_module.ExchangeError is ExchangeError
    assert not hasattr(exceptions_module, "ExchangerError")


@pytest.mark.fast_always
def test_breaking_api_cleanup_removes_transitional_public_surfaces() -> None:
    import vercor.grids as grids_module
    import vercor.output as output_module

    assert not hasattr(grids_module, "rectilinear")
    assert not hasattr(Coupler, "from_components")
    assert not hasattr(Coupler, "run_sequence")
    assert not hasattr(Coupler, "finalize")

    assert "rectilinear" not in grids_module.__all__

    for helper_name in (
        "output_masks_for_component",
        "write_coupler_component_snapshots",
        "write_coupler_runtime_outputs",
        "write_runtime_component_view_to_netcdf",
    ):
        assert helper_name not in output_module.__all__
        assert not hasattr(output_module, helper_name)

    with pytest.raises(TypeError, match="year_type"):
        cast(Any, Clock)(
            start=datetime(2000, 1, 1),
            dt_seconds=86400.0,
            steps=1,
            year_type="noleap",
        )
    assert not hasattr(Clock(datetime(2000, 1, 1), 60.0, 1), "year_type")

    with pytest.raises(TypeError, match="run_sequence"):
        cast(Any, Coupler)(
            clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            run_sequence=("ATM",),
        )


@pytest.mark.fast_always
def test_public_api_uses_canonical_breaking_names(
    tmp_path: Path,
) -> None:
    import vercor.grids as grids_module

    assert not hasattr(vercor, "SettingSpec")

    clock = Clock(
        start=datetime(2000, 1, 1),
        dt_seconds=86400.0,
        steps=1,
        calendar="360_day",
    )
    _, model_time, _ = next(clock.iter())
    assert isinstance(model_time, DateTime360)
    assert model_time.day_of_year == 1

    renamed_grid = grids_module.RectilinearGrid.uniform(
        "new-grid",
        nlon=2,
        nlat=2,
        longitude=(0.0, 90.0),
        latitude=(-45.0, 45.0),
    )

    component = DataComponent(
        "ATM",
        renamed_grid,
        fields={"temperature": 280.0},
        spec=ComponentSpec(outputs=("temperature", "humidity")),
    )
    assert component.spec.outputs == ("temperature", "humidity")

    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )
    assert coupler.run_order == ("ATM",)
    with pytest.raises(TypeError):
        coupler.components["NEW"] = component  # type: ignore[index]
    with pytest.raises(AttributeError):
        coupler.exchanges.append(Exchange("ATM", "ATM", ("temperature",)))  # type: ignore[attr-defined]

    state = coupler.initial_state()
    output_state = coupler.run(
        state,
        output=OutputTarget(
            tmp_path,
            write_period=False,
            write_final_fields=False,
            write_snapshots=False,
        ),
    )
    assert isinstance(output_state, RunState)
    assert not hasattr(Coupler, "write_outputs")


@pytest.mark.fast_always
def test_public_api_facade_exports_supported_names_only() -> None:
    assert tuple(vercor.__all__) == (
        "Clock",
        "Coupler",
        "Exchange",
        "RectilinearGrid",
        "RunState",
        "RuntimeOptions",
    )

    spec = ComponentSpec(
        inputs=("temperature", "temperature"),
        outputs=("sea_surface_temperature",),
        initial_fields={"sea_surface_temperature": 280.0},
    )
    assert spec.inputs == ("temperature",)
    assert spec.outputs == ("sea_surface_temperature",)
    assert spec.initial_fields == {"sea_surface_temperature": 280.0}

    result = StepResult(fields={"temperature": jnp.asarray(281.0)})
    assert tuple(result.fields) == ("temperature",)


@pytest.mark.fast_always
def test_step_result_payload_sentinel_preserves_runtime_payload_by_default() -> None:
    from vercor.components._runtime_fields import apply_step_result

    component = DataComponent(
        name="ATM",
        grid=make_test_grid(name="payload-sentinel"),
        fields={"temperature": 280.0},
    )
    payload = {"model": "state"}
    runtime_state = ComponentRuntimeState(
        fields=FieldStore.from_mapping({"temperature": jnp.asarray(280.0)}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
        payload=payload,
    )

    preserved = apply_step_result(
        component,
        runtime_state,
        StepResult(fields={"temperature": jnp.asarray(281.0)}),
    )
    cleared = apply_step_result(
        component,
        runtime_state,
        StepResult(fields={"temperature": jnp.asarray(282.0)}, payload=None),
    )

    assert preserved.payload is payload
    assert cleared.payload is None


@pytest.mark.fast_always
def test_exchange_accepts_supported_names_only() -> None:
    exchange = Exchange(
        "ATM",
        "OCN",
        ("temperature", vector("u_velocity", "v_velocity")),
        regridder_factory=bilinear,
    )

    assert exchange.source == "ATM"
    assert exchange.target == "OCN"
    assert exchange.fields == ("temperature", vector("u_velocity", "v_velocity"))
    assert exchange.regridder_factory is bilinear
    assert exchange.route_id == "ATM->OCN"


@pytest.mark.fast_always
def test_coupler_facade_wraps_runtime_state_and_views() -> None:
    component = DataComponent(
        name="ATM",
        grid=make_test_grid(name="coupler-reconfigured"),
        fields={"temperature": 280.0},
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    assert coupler.run_order == ("ATM",)
    assert coupler.run_order == coupler.run_order

    state = coupler.initial_state()
    view = state.component("ATM")
    views = state.components()

    assert views["ATM"] is view or views["ATM"].name == view.name
    assert view.field("temperature").shape == component.grid.shape


@pytest.mark.fast_always
def test_shallow_setup_regridding_grid_and_exchange_imports() -> None:
    from vercor.grids import RectilinearGrid as PublicRectilinearGrid
    from vercor.recipes import OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS
    from vercor.regridding import (
        bilinear as public_bilinear,
        conservative as public_conservative,
    )
    from vercor.setups import make_slab_ocean

    grid = PublicRectilinearGrid.uniform(
        "public-grid",
        nlon=4,
        nlat=3,
        longitude=(0.0, 360.0),
        latitude=(-90.0, 90.0),
    )

    assert isinstance(grid, PublicRectilinearGrid)
    assert public_bilinear is bilinear
    assert callable(public_conservative)
    assert OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS == ("sea_surface_temperature",)

    import vercor.exchanges as exchanges_module
    import vercor._regridders as regridders_module
    import vercor._regridders.bilinear as bilinear_module
    import vercor._regridders.conservative as conservative_module
    import vercor.regridding as regridding_module

    assert not hasattr(exchanges_module, "OCEAN_TO_ATMOSPHERE_SURFACE")
    for module, name in (
        (regridders_module, "BilinearRegridder"),
        (regridders_module, "BilinearRectilinearRegridder"),
        (regridders_module, "ConservativeRegridder"),
        (regridders_module, "ConservativeRectilinearRegridder"),
        (bilinear_module, "BilinearRegridder"),
        (conservative_module, "ConservativeRegridder"),
        (regridding_module, "BilinearRegridder"),
        (regridding_module, "BilinearRectilinearRegridder"),
        (regridding_module, "ConservativeRegridder"),
        (regridding_module, "ConservativeRectilinearRegridder"),
    ):
        assert not hasattr(module, name)
    assert hasattr(regridding_module, "Regridder")
    assert hasattr(regridding_module, "RegridderFactory")
    assert make_slab_ocean(grid).name == "OCN"
    assert make_slab_ocean(grid, mixed_layer_depth=30.0).name == "OCN"
    with pytest.raises(TypeError):
        make_slab_ocean(grid, H=30.0)  # type: ignore[call-arg]


@pytest.mark.fast_always
def test_top_level_exports_public_orchestration_and_component_author_api() -> None:
    expected_component_exports = (
        "CallableComponent",
        "Component",
        "ComponentSpec",
        "DataComponent",
        "LifecycleHooks",
        "PrefillContext",
        "PrefillResult",
        "SetupContext",
        "SetupResult",
        "StepContext",
        "StepResult",
        "TransferPolicy",
        "ValidationContext",
    )

    assert tuple(components_module.__all__) == expected_component_exports
    assert tuple(vercor.__all__) == (
        "Clock",
        "Coupler",
        "Exchange",
        "RectilinearGrid",
        "RunState",
        "RuntimeOptions",
    )

    assert components_module.Component is Component
    assert components_module.ComponentSpec is component_contracts_module.ComponentSpec
    assert components_module.SetupContext is component_contexts_module.SetupContext
    assert components_module.StepContext is component_contexts_module.StepContext
    assert components_module.StepResult is component_contracts_module.StepResult
    assert components_module.SetupResult is component_contracts_module.SetupResult
    assert components_module.TransferPolicy is component_contracts_module.TransferPolicy
    data_component_type = getattr(components_module, "DataComponent", None)
    assert data_component_type is not None
    assert components_module.DataComponent is data_component_type
    assert components_module.CallableComponent is base_module.CallableComponent


@pytest.mark.fast_always
def test_model_setup_factories_use_the_public_setup_owner() -> None:
    from vercor.setups._data.jcm_land import make_jcm_land
    from vercor.setups._external.camulator import make_camulator_gcm
    from vercor.setups._external.camulator_land import make_camulator_land
    from vercor.setups._external.jax_gcm import make_jax_gcm
    from vercor.setups._external.veros_gcm import make_veros_gcm
    from vercor.setups._jcm import make_jcm_land_atmosphere

    expected_factories = {
        "make_camulator_gcm": make_camulator_gcm,
        "make_camulator_land": make_camulator_land,
        "make_jax_gcm": make_jax_gcm,
        "make_jcm_land": make_jcm_land,
        "make_jcm_land_atmosphere": make_jcm_land_atmosphere,
        "make_veros_gcm": make_veros_gcm,
    }

    for name, factory in expected_factories.items():
        assert name in vercor.setups.__all__
        assert getattr(vercor.setups, name) is factory


@pytest.mark.fast_always
def test_components_package_exports_only_component_author_contracts() -> None:
    contracts_module = importlib.import_module("vercor.components.contracts")
    imported_data_module = importlib.import_module("vercor.components.data")

    assert base_module.__all__ == ["CallableComponent"]

    assert components_module.__all__ == [
        "CallableComponent",
        "Component",
        "ComponentSpec",
        "DataComponent",
        "LifecycleHooks",
        "PrefillContext",
        "PrefillResult",
        "SetupContext",
        "SetupResult",
        "StepContext",
        "StepResult",
        "TransferPolicy",
        "ValidationContext",
    ]
    assert components_module.Component is Component
    assert components_module.LifecycleHooks is contracts_module.LifecycleHooks
    assert data_module is imported_data_module
    assert components_module.DataComponent is data_module.DataComponent
    assert components_module.CallableComponent is base_module.CallableComponent
    assert components_module.ComponentSpec is contracts_module.ComponentSpec
    assert components_module.SetupContext is component_contexts_module.SetupContext
    assert components_module.StepContext is component_contexts_module.StepContext
    assert components_module.StepResult is contracts_module.StepResult
    assert components_module.SetupResult is contracts_module.SetupResult
    assert components_module.TransferPolicy is contracts_module.TransferPolicy

    assert setup_validation_module.validate_component_setup is not None


@pytest.mark.fast_always
def test_component_base_internals_are_private_modules() -> None:
    base_source = Path("vercor/components/base.py").read_text(encoding="utf-8")
    contracts_source = Path("vercor/components/contracts.py").read_text(
        encoding="utf-8"
    )
    protocol_source = Path("vercor/components/_protocol.py").read_text(encoding="utf-8")
    data_source = Path("vercor/components/data.py").read_text(encoding="utf-8")
    adapter_source = Path("vercor/components/_adapter.py").read_text(encoding="utf-8")

    assert "class Component(Protocol)" in protocol_source
    assert "from vercor.components._protocol import (" in contracts_source
    assert "class CallableComponent" in base_source
    assert "class DataComponent" in data_source
    assert "class _ComponentBinding" in adapter_source
    assert hasattr(Component, "step")


@pytest.mark.fast_always
def test_runtime_field_state_only_helpers_do_not_accept_unused_component() -> None:
    runtime_fields_module = importlib.import_module("vercor.components._runtime_fields")

    assert tuple(signature(runtime_fields_module.runtime_fields).parameters) == (
        "component_state",
    )


@pytest.mark.fast_always
def test_component_contract_modules_share_field_name_deduplication_owner() -> None:
    field_names_module = importlib.import_module("vercor._field_names")
    private_contracts_module = importlib.import_module("vercor.components._contracts")

    assert field_names_module.unique_field_names(("a", "b", "a")) == ("a", "b")
    assert (
        component_contracts_module._unique_field_names
        is field_names_module.unique_field_names
    )
    assert private_contracts_module.unique_field_names is (
        field_names_module.unique_field_names
    )

    contracts_source = Path("vercor/components/contracts.py").read_text(
        encoding="utf-8"
    )
    private_contracts_source = Path("vercor/components/_contracts.py").read_text(
        encoding="utf-8"
    )
    assert "def _unique_field_names(" not in contracts_source
    assert "def unique_field_names(" not in private_contracts_source
    assert "vercor._field_names" in contracts_source
    assert "vercor._field_names" in private_contracts_source


@pytest.mark.fast_always
def test_runtime_component_type_imports_are_annotation_only() -> None:
    """Private runtime modules should import Component only for type checking."""

    modules_with_annotation_only_component_usage = (
        Path("vercor/_runtime/initialization.py"),
        Path("vercor/_runtime/topology.py"),
        Path("vercor/_runtime/coupler_state.py"),
    )

    for path in modules_with_annotation_only_component_usage:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "vercor.components.base"
            ):
                imported_names = {alias.name for alias in node.names}
                assert "Component" not in imported_names, path
        assert "if TYPE_CHECKING:" in source, path

    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")
    assert "import vercor.components as _components" in coupler_source
    assert "Iterable[_components.Component]" in coupler_source
    assert "from vercor.components import Component" not in coupler_source
    assert "_ComponentInfo" not in coupler_source


@pytest.mark.fast_always
def test_setup_components_do_not_expose_mutable_metadata_mapping() -> None:
    component = DataComponent(
        name="ATM",
        grid=make_test_grid(name="metadata-boundary"),
    )

    assert not hasattr(component, "setup_metadata")
    assert not hasattr(component, "_setup_metadata")

    helper_source = Path("vercor/setups/_data/_component_helpers.py").read_text(
        encoding="utf-8"
    )
    era5_atmosphere_source = Path("vercor/setups/_data/era5_atmosphere.py").read_text(
        encoding="utf-8"
    )

    assert "cast(Any, component).DATA_FILES" not in helper_source
    assert "cast(Any, component).hyai" not in era5_atmosphere_source
    assert "cast(Any, component).hybi" not in era5_atmosphere_source
    assert "cast(Any, component).hyam" not in era5_atmosphere_source
    assert "cast(Any, component).hybm" not in era5_atmosphere_source
    assert "SetupResult(" in era5_atmosphere_source


@pytest.mark.fast_always
def test_coupler_run_order_is_explicit_empty_schedule_by_default() -> None:
    coupler = vercor.Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1)
    )

    assert coupler.run_order == ()

    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")
    assert 'hasattr(self, "run_order")' not in coupler_source
    assert 'getattr(self, "run_order"' not in coupler_source


@pytest.mark.fast_always
def test_coupler_accepts_plain_component_name_sequences() -> None:
    import numpy as np

    from tests._coverage_support import make_test_grid as _make_grid

    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1)
    grid = _make_grid("grid")
    ocean = DataComponent(
        "OCN",
        grid,
        fields={"sea_surface_temperature": np.zeros(grid.shape)},
    )
    atmosphere = DataComponent(
        "ATM",
        grid,
        fields={"sea_surface_temperature": np.zeros(grid.shape)},
    )

    coupler = Coupler(
        clock=clock,
        components=(ocean, atmosphere),
        run_order=["OCN", "ATM"],
    )

    assert coupler.run_order == ("OCN", "ATM")

    assert not hasattr(coupler, "set_components_run_order")
    assert not hasattr(coupler, "set_run_order")


@pytest.mark.fast_always
def test_coupler_rejects_string_run_order() -> None:
    with pytest.raises(
        CouplerError,
        match="run_order must be a sequence of component names",
    ):
        Coupler(
            clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            run_order="ATM",
        )


@pytest.mark.fast_always
def test_setup_state_reads_run_order_as_plain_sequence() -> None:
    setup_state_paths = (
        Path("vercor/setups/_external/jax_gcm_state.py"),
        Path("vercor/setups/_external/veros_gcm_state.py"),
    )

    for path in setup_state_paths:
        assert ".run_order.order" not in path.read_text(encoding="utf-8")


@pytest.mark.fast_always
def test_multi_exchange_setup_scripts_use_constructor_assembly() -> None:
    multi_exchange_scripts = (
        Path("vercor/setups/gallery/run_data_driver.py"),
        Path("vercor/setups/gallery/run_jcm_with_verosdata.py"),
        Path("vercor/setups/gallery/run_jcm_with_veros.py"),
        Path("vercor/setups/gallery/run_jcm_with_slab.py"),
        Path("vercor/setups/gallery/run_slab_driver.py"),
    )

    for path in multi_exchange_scripts:
        source = path.read_text(encoding="utf-8")
        assert ".add_exchanges(" not in source, path
        assert "add_exchange_specs" not in source, path
        assert "cpl.add_exchange(" not in source, path


@pytest.mark.fast_always
def test_slab_driver_uses_runtime_views_for_ice_diagnostics() -> None:
    slab_source = Path("vercor/setups/gallery/run_slab_driver.py").read_text(
        encoding="utf-8"
    )

    assert 'final_state.components(("ATM", "OCN", "LND", "ICE"))' in slab_source
    assert 'views["ICE"].field("ice_fraction")' in slab_source
    assert 'get_component_state("ICE").fields.get("ice_fraction")' not in slab_source


@pytest.mark.fast_always
def test_runtime_state_is_separate_from_public_component_objects() -> None:
    assert hasattr(components_module, "DataComponent")
    component = components_module.DataComponent(
        name="ATM",
        grid=make_test_grid(name="api-boundary"),
    )
    runtime_state = ComponentRuntimeState(
        fields=FieldStore.empty(),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )

    assert not isinstance(runtime_state, Component)
    assert not hasattr(runtime_state, "name")
    assert not hasattr(runtime_state, "grid")
    assert not hasattr(runtime_state, "settings")
    assert not hasattr(component, "incoming")
    assert not hasattr(component, "outgoing")
    assert not hasattr(component, "with_data")


@pytest.mark.fast_always
def test_setup_factories_are_primary_concrete_component_api() -> None:
    from vercor.setups._slab import (
        make_slab_atmosphere,
        make_slab_land,
        make_slab_ocean,
        make_slab_seaice,
    )

    grid = make_test_grid(name="setup-api")
    assert isinstance(make_slab_atmosphere(grid), Component)
    assert isinstance(make_slab_ocean(grid), Component)
    assert isinstance(make_slab_land(grid), Component)
    assert isinstance(make_slab_seaice(grid), Component)

    from vercor.setups._data.era5_land import make_era5_land
    from vercor.setups._external.veros_gcm import make_veros_gcm

    assert callable(make_era5_land)
    assert callable(make_veros_gcm)


@pytest.mark.fast_always
def test_setup_modules_do_not_subclass_component_contracts() -> None:
    forbidden_bases = {
        "Component",
        "DataComponent",
        "HostComponent",
        "HostRuntimeComponent",
    }
    for path in Path("vercor/setups").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = {
                    base.id for base in node.bases if isinstance(base, ast.Name)
                } | {
                    base.attr for base in node.bases if isinstance(base, ast.Attribute)
                }
                assert forbidden_bases.isdisjoint(
                    base_names
                ), f"{path}:{node.lineno} subclasses a core component contract"


@pytest.mark.fast_always
def test_private_setup_state_objects_do_not_borrow_component_methods() -> None:
    forbidden_markers = (
        "grid_field_defaults = Component.",
        "seed_field = Component.",
        "seed_fields = Component.",
        "prefill_runtime_fields = Component.",
    )
    for path in (
        Path("vercor/setups/_external/jax_gcm.py"),
        Path("vercor/setups/_external/veros_gcm.py"),
        Path("vercor/setups/_external/camulator.py"),
        Path("vercor/setups/_external/camulator_land.py"),
    ):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in source, f"{path} borrows {marker}"


@pytest.mark.fast_always
def test_setup_adapters_do_not_import_runtime_context_or_store_internals() -> None:
    forbidden_markers = (
        "vercor._runtime.contexts",
        "from vercor._runtime.contexts import ComponentInitContext",
        "from vercor._runtime.contexts import RuntimeStepContext",
        "from vercor._runtime import FieldStore",
        "FieldStore.from_mapping",
    )
    for path in Path("vercor/setups").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in source, f"{path} imports runtime internals"


@pytest.mark.fast_always
def test_shared_helpers_have_core_owners_not_setup_or_regridder_owners() -> None:
    import vercor.calendar as calendar_module
    import vercor.forcing_index as forcing_index_module
    import vercor.fluxes.vertical_coordinates as vertical_module
    import vercor.grid_geometry as grid_geometry_module
    import vercor.physics as physics_module
    import vercor.time_selection as time_selection_module

    jax_gcm_pytree_module = importlib.import_module(
        "vercor.setups._external._jax_gcm_pytree"
    )

    assert callable(calendar_module.is_leap_year)
    assert callable(forcing_index_module.daily_forcing_index)
    assert callable(grid_geometry_module.centers_to_edges)
    assert callable(vertical_module.compute_sigma_pressure_levels)
    assert callable(vertical_module.compute_hybrid_pressure_levels)
    assert callable(vertical_module.compute_hybrid_sigma_full_level_altitudes)
    assert callable(vertical_module.get_altitudes_sigma_levels)
    assert callable(jax_gcm_pytree_module.tree_as_real_dtype)
    assert callable(jax_gcm_pytree_module.tree_mean)
    assert callable(jax_gcm_pytree_module.tree_stack)
    assert callable(jax_gcm_pytree_module.tree_unwrap_leading_dims)
    assert callable(time_selection_module.datetime_to_seconds_in_year)
    assert callable(time_selection_module.get_periodic_interval)
    assert "gravity" in physics_module.PhysicalConstants.__dataclass_fields__
    assert "sea_surface_temperature" in COMMON_FIELD_NAMES
    assert calendar_module.is_leap_year.__module__ == "vercor.calendar"
    assert forcing_index_module.daily_forcing_index.__module__ == (
        "vercor.forcing_index"
    )
    assert grid_geometry_module.centers_to_edges.__module__ == "vercor.grid_geometry"
    assert time_selection_module.get_periodic_interval.__module__ == (
        "vercor.time_selection"
    )


@pytest.mark.fast_always
def test_setup_lazy_exports_have_one_public_registry() -> None:
    import vercor.setups as setups_module
    import vercor.setups._data as data_setups_module
    import vercor.setups._external as external_setups_module

    assert {"JCMInputs", "load_jcm_inputs"}.issubset(set(setups_module.__all__))

    assert hasattr(setups_module, "_LAZY_EXPORTS")
    for private_package in (data_setups_module, external_setups_module):
        assert private_package.__all__ == []
        assert not hasattr(private_package, "_LAZY_EXPORTS")
        assert "__getattr__" not in vars(private_package)


@pytest.mark.fast_always
def test_callable_component_has_one_step_normalization_owner() -> None:
    callable_source = Path("vercor/components/_callable_wrappers.py").read_text(
        encoding="utf-8"
    )
    base_source = Path("vercor/components/base.py").read_text(encoding="utf-8")

    assert "def normalize_component_step_callable(" in callable_source
    assert base_source.count("normalize_component_step_callable(") == 1


@pytest.mark.fast_always
def test_production_runtime_modules_use_coupler_state_name() -> None:
    production_paths = (
        Path("vercor/_runtime/coupler_state.py"),
        Path("vercor/_runtime/driver.py"),
        Path("vercor/_runtime/exchange_dispatch.py"),
        Path("vercor/_runtime/facade.py"),
        Path("vercor/_runtime/preparation.py"),
        Path("vercor/_runtime/state_validation.py"),
        Path("vercor/output/_runtime.py"),
    )

    for path in production_paths:
        source = path.read_text(encoding="utf-8")
        assert "RuntimeCouplerState" not in source, path


@pytest.mark.fast_always
def test_shared_base_owns_scalar_regrid_dispatch() -> None:
    regridder_base_source = Path("vercor/_regridders/base.py").read_text(
        encoding="utf-8"
    )
    bilinear_source = Path("vercor/_regridders/bilinear.py").read_text(encoding="utf-8")
    conservative_source = Path("vercor/_regridders/conservative.py").read_text(
        encoding="utf-8"
    )

    assert "def __call__(" not in regridder_base_source
    assert "def regrid(" in regridder_base_source
    assert "def regrid_vector(" not in regridder_base_source
    assert "def _ensure_ready(" not in regridder_base_source
    assert "def __call__(" not in bilinear_source
    assert "def regrid(" not in bilinear_source
    assert "def regrid_vector(" in bilinear_source
    assert "apply_vector" in bilinear_source
    assert "def __call__(" not in conservative_source
    assert "def regrid(" not in conservative_source
    assert "def regrid_vector(" not in conservative_source
    assert "def _ensure_ready(" not in conservative_source
    assert "apply_vector" not in conservative_source


@pytest.mark.fast_always
def test_setup_helper_and_external_output_ownership_boundaries() -> None:
    output_init_source = Path("vercor/output/__init__.py").read_text(encoding="utf-8")
    output_session_source = Path("vercor/output/_session.py").read_text(
        encoding="utf-8"
    )
    native_sources = tuple(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "vercor/setups/_external/jax_gcm_output.py",
            "vercor/setups/_external/veros_output.py",
            "vercor/setups/_external/camulator_output.py",
        )
    )

    assert not Path("vercor/output/_component_adapter.py").exists()
    assert not Path("vercor/output/_period_files.py").exists()
    assert "class PeriodAverageAccumulator" not in Path(
        "vercor/output/_period.py"
    ).read_text(encoding="utf-8")
    assert output_session_source.count("class _OutputAccumulator") == 1
    assert "class OutputSpec" in output_init_source
    assert '"OutputSpec"' in output_init_source
    assert "class OutputConfig" not in output_init_source
    for source in native_sources:
        assert "def sample(self, context: OutputContext) -> OutputFrame:" in source
        assert "record_period" not in source


@pytest.mark.fast_always
def test_external_adapter_all_exports_are_public() -> None:
    for path in Path("vercor/setups/_external").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                continue
            exports = [
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            private_exports = [name for name in exports if name.startswith("_")]
            assert (
                private_exports == []
            ), f"{path} exports private names: {private_exports}"


@pytest.mark.fast_always
def test_external_runtime_helpers_use_concrete_setup_state_annotations() -> None:
    runtime_sources = {
        "vercor/setups/_external/jax_gcm_runtime.py": (
            "JAXGCMSetupState",
            "vercor.setups._external.jax_gcm_state",
            "state",
        ),
        "vercor/setups/_external/veros_runtime.py": (
            "VerosGCMSetupState",
            "vercor.setups._external.veros_gcm_state",
            "resources",
        ),
        "vercor/setups/_external/camulator_runtime.py": (
            "CAMulatorGCMSetupState",
            "vercor.setups._external.camulator_gcm_state",
            "resources",
        ),
    }

    for path_name, (
        state_name,
        state_module,
        parameter_name,
    ) in runtime_sources.items():
        source = Path(path_name).read_text(encoding="utf-8")
        assert "Protocol" not in source
        assert f"from {state_module} import {state_name}" in source
        assert f"{parameter_name}: {state_name}" in source or (
            f'{parameter_name}: "{state_name}"' in source
        )


@pytest.mark.fast_always
def test_jax_gcm_factory_binds_runtime_hooks_directly() -> None:
    source = Path("vercor/setups/_external/jax_gcm.py").read_text(encoding="utf-8")
    factory_source = source.split("def make_jax_gcm(", 1)[1]

    assert "from functools import partial" in source
    assert "partial(_jax_gcm_runtime.step_jax_gcm_component, state)" in factory_source
    assert "setup=state.setup" in factory_source
    assert "_jax_gcm_runtime.prefill_jax_gcm_runtime_fields" in factory_source
    assert "_jax_gcm_runtime.validate_jax_gcm_runtime_state" in factory_source


@pytest.mark.fast_always
def test_jax_gcm_adapter_uses_jcm_2_api_owners() -> None:
    state_source = Path("vercor/setups/_external/jax_gcm_state.py").read_text(
        encoding="utf-8"
    )
    factory_source = Path("vercor/setups/_external/jax_gcm.py").read_text(
        encoding="utf-8"
    )

    assert "from jcm.physics.speedy.speedy_terms import speedy_physics" in state_source
    assert "from jcm.terrain import TerrainData" in state_source
    assert "from jcm.forcing import ForcingData" in state_source
    assert "SpeedyPhysics" not in state_source
    assert "dynamics_state_to_physics_state" not in state_source
    assert "_prepare_initial_modal_state" not in state_source
    assert "from jcm.terrain import TerrainData as _TerrainData" in factory_source


@pytest.mark.fast_always
def test_jax_gcm_average_writer_bypasses_xarray_adapter() -> None:
    source = Path("vercor/setups/_external/jax_gcm_output.py").read_text(
        encoding="utf-8"
    )

    assert "import xarray" not in source
    assert ".to_xarray(" not in source


@pytest.mark.fast_always
def test_external_package_has_no_top_level_import_cycles() -> None:
    assert (
        package_import_cycles("vercor/setups/_external", "vercor.setups._external")
        == []
    )


@pytest.mark.fast_always
def test_output_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/output", "vercor.output") == []


@pytest.mark.fast_always
def test_veros_runtime_settings_imports_runtime_settings_lazily() -> None:
    source = Path("vercor/setups/_external/veros_runtime_settings.py").read_text(
        encoding="utf-8"
    )
    before_function, function_body = source.split(
        "def configure_veros_runtime() -> None:",
        1,
    )

    assert "from veros import runtime_settings" not in before_function
    assert "from veros import runtime_settings" in function_body


@pytest.mark.fast_always
def test_common_exchange_recipes_are_centralized_for_examples() -> None:
    import vercor.recipes as recipes_module
    from vercor.recipes import (
        ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
        JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS,
        SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS,
    )

    required_recipes = (
        "ATMOSPHERE_TO_DATA_OCEAN_FIELDS",
        "ATMOSPHERE_TO_LAND_RADIATION_FIELDS",
        "ATMOSPHERE_TO_LAND_STATE_FIELDS",
        "JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS",
        "LAND_TO_ATMOSPHERE_SURFACE_FIELDS",
        "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS",
        "SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS",
        "SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS",
    )
    recipes_source = Path("vercor/recipes.py").read_text(encoding="utf-8")
    for recipe_name in required_recipes:
        assert recipe_name in recipes_module.__all__
        assert isinstance(getattr(recipes_module, recipe_name), tuple)
        assert f"{recipe_name}: tuple[_ExchangeField, ...]" in recipes_source

    recipe_users = (
        Path("vercor/setups/gallery/run_jcm_with_verosdata.py"),
        Path("vercor/setups/gallery/run_jcm_with_era5data.py"),
        Path("vercor/setups/gallery/run_jcm_with_slab.py"),
        Path("vercor/setups/gallery/run_camulator_with_veros.py"),
        Path("vercor/setups/gallery/run_data_driver.py"),
        Path("vercor/setups/gallery/run_veros_with_era5data.py"),
        Path("vercor/setups/gallery/profile_runtime.py"),
    )
    for path in recipe_users:
        source = path.read_text(encoding="utf-8")
        assert "from vercor.recipes import" in source, path
        if path.name.startswith("run_"):
            assert "Exchange(" in source, path

    assert _flatten_field_items(
        JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS
    ) == _flatten_field_items(
        (*ATMOSPHERE_TO_DATA_OCEAN_FIELDS, *SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS)
    )
    jcm_slab_source = Path("vercor/setups/gallery/run_jcm_with_slab.py").read_text(
        encoding="utf-8"
    )
    assert "JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS" in jcm_slab_source


@pytest.mark.fast_always
def test_assets_and_diagnostics_have_focused_ownership_boundaries() -> None:
    import vercor.assets as assets_module
    import vercor.diagnostics as diagnostics_module
    import vercor.setups._data.assets as setup_assets_module

    assert assets_module.ensure_registered_asset.__module__ == "vercor.assets"
    assert setup_assets_module.get_forcing_data.__module__ == (
        "vercor.setups._data.assets"
    )

    assert diagnostics_module.combine_surface_temperatures is not None
    assert diagnostics_module.print_component_field_means_table is not None
    assert diagnostics_module.plot_component_scalar_vector_comparison is not None
    assert diagnostics_module.combine_surface_temperatures.__module__ == (
        "vercor.diagnostics.fields"
    )


@pytest.mark.fast_always
def test_veros_factory_binds_current_runtime_step_and_setup() -> None:
    gcm_source = Path("vercor/setups/_external/veros_gcm.py").read_text(
        encoding="utf-8"
    )
    assert "from functools import partial" in gcm_source
    loader_source, factory_source = gcm_source.split("def make_veros_gcm(", 1)
    assert (
        "import vercor.setups._external.veros_runtime as veros_runtime" in loader_source
    )
    assert "configure_veros_runtime()" in factory_source
    assert "_load_veros_implementation()" in factory_source
    assert factory_source.index("configure_veros_runtime()") < factory_source.index(
        "_load_veros_implementation()"
    )
    assert "partial(_veros_runtime.step_veros_runtime, state)" in factory_source
    assert "LifecycleHooks(setup=state.setup)" in factory_source
    assert "_veros_output.veros_output_provider()" in factory_source
    assert "partial(_veros_output.write_veros_snapshot_output, state)" not in (
        factory_source
    )
    runtime_source = Path("vercor/setups/_external/veros_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "resources._veros_state" not in runtime_source


@pytest.mark.fast_always
def test_camulator_adapters_share_runtime_cursor_state_transition_helper() -> None:
    for path in (Path("vercor/setups/_external/camulator_gcm_state.py"),):
        source = path.read_text(encoding="utf-8")
        assert "CamulatorRuntimeCursor" in source, path
        assert "runtime_forcing_index(" not in source, path
        assert "timestep_counter += 1" not in source, path

    for path in (
        Path("vercor/setups/_external/camulator.py"),
        Path("vercor/setups/_external/camulator_land.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "runtime_forcing_index(" not in source, path
        assert "timestep_counter += 1" not in source, path


@pytest.mark.fast_always
def test_camulator_gcm_factory_passes_runtime_step_directly() -> None:
    gcm_source = Path("vercor/setups/_external/camulator.py").read_text(
        encoding="utf-8"
    )
    state_source = Path("vercor/setups/_external/camulator_gcm_state.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(state_source)
    setup_state_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CAMulatorGCMSetupState"
    )

    assert "step" not in {
        node.name
        for node in setup_state_class.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "from functools import partial" in gcm_source
    assert (
        "import vercor.setups._external.camulator_runtime as _camulator_runtime"
        in gcm_source
    )
    assert "partial(_camulator_runtime.step_camulator_runtime, resources)" in gcm_source
    assert "LifecycleHooks(setup=resources.setup)" in gcm_source
    setup_attributes = {
        target.attr
        for node in ast.walk(setup_state_class)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert setup_attributes.isdisjoint(
        {
            "state",
            "runtime_cursor",
            "_output_prediction",
            "_output_prediction_samples",
            "forecast_hour",
        }
    )


@pytest.mark.fast_always
def test_jcm_land_inlines_single_use_coordinate_conversion() -> None:
    source = Path("vercor/setups/_data/jcm_land.py").read_text(encoding="utf-8")

    assert "def _jcm_coordinates_in_degrees" not in source
    assert "def _prepare_jcm_land_runtime_fields" not in source
    assert "jnp.rad2deg(" in source
    assert "def _coordinates_in_degrees" not in source


@pytest.mark.fast_always
def test_bilinear_interpolator_removes_unused_cartesian_helper() -> None:
    source = Path("vercor/_interpolators/bilinear_rectilinear.py").read_text(
        encoding="utf-8"
    )

    assert "def _geo_to_cart(" not in source
    assert "_lon_src_2d" not in source
    assert "_lat_src_2d" not in source


@pytest.mark.fast_always
def test_jax_gcm_factory_does_not_attach_test_only_setup_state() -> None:
    jax_gcm_source = Path("vercor/setups/_external/jax_gcm.py").read_text(
        encoding="utf-8"
    )
    runtime_test_source = Path("tests/test_coupler_runtime.py").read_text(
        encoding="utf-8"
    )

    forbidden_factory_markers = (
        "component_any = cast(Any, component)",
        "component_any.model =",
        "component_any.sigma_levels =",
        "component_any._setup_state =",
    )
    for marker in forbidden_factory_markers:
        assert marker not in jax_gcm_source

    assert "_setup_state" not in runtime_test_source

    jcm_slab_source = Path("vercor/setups/gallery/run_jcm_with_slab.py").read_text(
        encoding="utf-8"
    )
    assert 'getattr(atm, "model")' not in jcm_slab_source


@pytest.mark.fast_always
def test_jcm_examples_use_public_input_loader_facade() -> None:
    jcm_slab_source = Path("vercor/setups/gallery/run_jcm_with_slab.py").read_text(
        encoding="utf-8"
    )
    assert "vercor.setups._external.jax_gcm_tools" not in jcm_slab_source
    assert "load_jcm_inputs" in jcm_slab_source

    jcm_veros_source = Path("vercor/setups/gallery/run_jcm_with_veros.py").read_text(
        encoding="utf-8"
    )
    assert "generate_jcm_coords_forcing_topography_files" not in jcm_veros_source


@pytest.mark.fast_always
def test_data_and_callable_factories_return_core_contract_instances() -> None:
    from vercor.setups._data.era5_land import make_era5_land
    from vercor.setups._external.camulator import make_camulator_gcm

    assert callable(make_era5_land)
    assert callable(make_camulator_gcm)
    grid = make_test_grid(name="component-protocol-instances")
    assert isinstance(DataComponent("DATA", grid), Component)
    assert isinstance(CallableComponent("CALLABLE", grid, lambda fields: {}), Component)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "import_statement",
    (
        "import vercor.setups._external",
        "import vercor.setups._data",
        "import vercor.setups._jcm",
        "from vercor.setups._external import __all__",
        "from vercor.setups._data import __all__",
        "import vercor.setups._data.era5_land",
    ),
)
def test_unrelated_setup_imports_do_not_initialize_optional_adapters(
    import_statement: str,
) -> None:
    module_probe = (
        "import sys\n"
        f"statement = {import_statement!r}\n"
        "exec(statement)\n"
        "heavy = {'torch', 'jcm', 'dinosaur', 'veros'} & set(sys.modules)\n"
        "assert not heavy, sorted(heavy)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", module_probe],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    output = completed.stdout + completed.stderr
    forbidden_markers = (
        "CREDIT modules not fully available",
        "credit.postblock not available",
        "Credit module not found",
        "Importing core modules",
        "Using computational backend",
        "Runtime settings are now locked",
    )
    for marker in forbidden_markers:
        assert marker not in output
