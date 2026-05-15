from __future__ import annotations

from datetime import datetime
from inspect import signature
from pathlib import Path
import ast
import subprocess
import sys

import pytest

import vercor
import vercor.components as components_module
from tests._coverage_support import make_test_grid
from vercor.components.base import Component, DataComponent, HostRuntimeComponent
from vercor.clock import Clock
from vercor.exchange import Exchange
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.run_sequence import RunSequence
from vercor.runtime import RuntimeComponentState, RuntimeFieldStore
from vercor.regridders import bilinear


@pytest.mark.fast_always
def test_top_level_exports_public_orchestration_and_component_author_api() -> None:
    expected_public_names = {
        "Clock",
        "Component",
        "ComponentFieldSpec",
        "ComponentSetupContext",
        "ComponentStepContext",
        "ComponentStepResult",
        "Coupler",
        "DataComponent",
        "Exchange",
        "HostRuntimeComponent",
        "RectilinearGrid",
        "RunSequence",
        "data_component",
        "differentiable_component",
        "host_component",
    }
    runtime_internal_names = {
        "ComponentInitContext",
        "RuntimeComponentContract",
        "RuntimeComponentState",
        "RuntimeComponentView",
        "RuntimeCouplerState",
        "RuntimeDispatchContext",
        "RuntimeFieldStore",
        "RuntimeStepContext",
        "RuntimeStepInfo",
    }
    legacy_component_names = {
        "make_data_component",
        "make_differentiable_component",
        "make_host_component",
    }

    assert expected_public_names.issubset(set(vercor.__all__))
    assert runtime_internal_names.isdisjoint(set(vercor.__all__))
    assert legacy_component_names.isdisjoint(set(vercor.__all__))

    assert vercor.Component is Component
    assert vercor.ComponentFieldSpec is components_module.ComponentFieldSpec
    assert vercor.ComponentSetupContext is ComponentInitContext
    assert vercor.ComponentStepContext is RuntimeStepContext
    assert vercor.ComponentStepResult is components_module.ComponentStepResult
    data_component_type = getattr(components_module, "DataComponent", None)
    assert data_component_type is not None
    assert getattr(vercor, "DataComponent", None) is data_component_type
    assert vercor.HostRuntimeComponent is HostRuntimeComponent
    assert vercor.data_component is components_module.data_component
    assert vercor.differentiable_component is components_module.differentiable_component
    assert vercor.host_component is components_module.host_component
    assert vercor.RunSequence is RunSequence
    for name in (*runtime_internal_names, *legacy_component_names):
        assert not hasattr(vercor, name)


@pytest.mark.fast_always
def test_components_package_exports_only_component_author_contracts() -> None:
    assert components_module.__all__ == [
        "Component",
        "ComponentFieldSpec",
        "ComponentSetupContext",
        "ComponentStepContext",
        "ComponentStepResult",
        "DataComponent",
        "HostRuntimeComponent",
        "data_component",
        "differentiable_component",
        "host_component",
    ]
    assert components_module.Component is Component
    assert hasattr(components_module, "ComponentFieldSpec")
    assert components_module.ComponentSetupContext is ComponentInitContext
    assert components_module.ComponentStepContext is RuntimeStepContext
    assert hasattr(components_module, "ComponentStepResult")
    assert hasattr(components_module, "DataComponent")
    assert components_module.HostRuntimeComponent is HostRuntimeComponent
    assert hasattr(components_module, "data_component")
    assert hasattr(components_module, "differentiable_component")
    assert hasattr(components_module, "host_component")
    assert not hasattr(components_module, "make_data_component")
    assert not hasattr(components_module, "make_differentiable_component")
    assert not hasattr(components_module, "make_host_component")
    assert not hasattr(components_module, "RuntimeComponentState")
    assert not hasattr(components_module, "ComponentInitContext")
    assert not hasattr(components_module, "RuntimeStepContext")


@pytest.mark.fast_always
def test_callable_author_api_does_not_expose_legacy_field_seed_keyword() -> None:
    public_callables = (
        components_module.Component.from_model,
        components_module.HostRuntimeComponent.from_model,
        components_module.differentiable_component,
        components_module.host_component,
    )
    removed_keyword = "initial" + "_fields"

    for callable_factory in public_callables:
        parameters = signature(callable_factory).parameters
        assert removed_keyword not in parameters
        assert "required_fields" not in parameters
        assert parameters["payload"].kind is parameters["payload"].KEYWORD_ONLY
        assert parameters["settings"].kind is parameters["settings"].KEYWORD_ONLY


@pytest.mark.fast_always
def test_component_base_internals_are_private_modules() -> None:
    base_source = Path("vercor/components/base.py").read_text(encoding="utf-8")
    contracts_source = Path("vercor/components/_contracts.py").read_text(
        encoding="utf-8"
    )
    callable_source = Path("vercor/components/_callable_wrappers.py").read_text(
        encoding="utf-8"
    )
    runtime_fields_source = Path("vercor/components/_runtime_fields.py").read_text(
        encoding="utf-8"
    )
    validation_source = Path("vercor/components/_validation.py").read_text(
        encoding="utf-8"
    )

    assert "class ComponentFieldSpec" in contracts_source
    assert "class ComponentStepResult" in contracts_source
    assert "def normalize_author_field_values" in contracts_source
    assert "class _CallableRuntimeMixin" in callable_source
    assert "def normalize_component_step_callable" in callable_source
    assert "def runtime_fields(" in runtime_fields_source
    assert "def runtime_field(" in runtime_fields_source
    assert "def runtime_field_or(" in runtime_fields_source
    assert "def runtime_field_or_zeros_like(" in runtime_fields_source
    assert "def with_runtime_fields(" in runtime_fields_source
    assert "def apply_step_result(" in runtime_fields_source
    assert "def prefill_runtime_fields(" in runtime_fields_source
    assert "def require_runtime_fields(" in runtime_fields_source
    assert "def validate_declared_runtime_fields(" in runtime_fields_source
    assert "def validate_component_setup" in validation_source
    assert "def _author_field_spec(" not in base_source
    assert "def component_field_spec(" not in contracts_source
    assert "def _callable_component_from_model(" in base_source
    assert base_source.count("_callable_component_from_model(") == 3
    assert base_source.count("_create_callable_component(") == 1
    assert "_required_fields" not in callable_source
    assert "_prefill_fields" not in callable_source
    assert "_field_defaults" not in callable_source
    assert "required_fields:" not in callable_source
    assert "prefill_fields:" not in callable_source
    assert "field_defaults:" not in callable_source
    assert "def apply_callable_step_result" not in callable_source
    assert "def make_callable_component" not in callable_source
    assert "def make_callable_host_component" not in callable_source
    assert "def _create_callable_component" in callable_source

    private_markers = (
        "class _CallableRuntimeMixin",
        "class _CallableComponent",
        "class _CallableHostRuntimeComponent",
        "def _normalize_component_step_callable",
        "def _component_step_signature_error",
        "def _make_differentiable_callable_component",
        "def _make_host_callable_component",
        "def make_callable_component",
        "def make_callable_host_component",
        "def make_data_component",
        "def make_differentiable_component",
        "def make_host_component",
        "component_state.data.to_mapping()",
        "component_state.data.replace_many(fields)",
        "validate_runtime_component_data_field",
    )
    for marker in private_markers:
        assert marker not in base_source

    assert "_contracts" not in components_module.__all__
    assert "_callable_wrappers" not in components_module.__all__
    assert "_runtime_fields" not in components_module.__all__
    assert "_validation" not in components_module.__all__


@pytest.mark.fast_always
def test_setup_components_use_explicit_metadata_mapping() -> None:
    component = DataComponent(
        name="ATM",
        grid=make_test_grid(name="metadata-boundary"),
    )

    component.setup_metadata["DATA_FILES"] = {"surface": "surface.nc"}

    assert component.setup_metadata["DATA_FILES"] == {"surface": "surface.nc"}

    helper_source = Path("setups/data/_component_helpers.py").read_text(
        encoding="utf-8"
    )
    era5_atmosphere_source = Path("setups/data/era5_atmosphere.py").read_text(
        encoding="utf-8"
    )

    assert "cast(Any, component).DATA_FILES" not in helper_source
    assert "cast(Any, component).hyai" not in era5_atmosphere_source
    assert "cast(Any, component).hybi" not in era5_atmosphere_source
    assert "cast(Any, component).hyam" not in era5_atmosphere_source
    assert "cast(Any, component).hybm" not in era5_atmosphere_source


@pytest.mark.fast_always
def test_setup_forcing_reader_reexports_canonical_read_boundary() -> None:
    import setups.data.forcing as setup_forcing_module
    import vercor.forcing_data as forcing_data_module

    forcing_source = Path("setups/data/forcing.py").read_text(encoding="utf-8")

    assert setup_forcing_module.read_forcing is forcing_data_module.read_forcing
    assert "def read_forcing(" not in forcing_source


@pytest.mark.fast_always
def test_setup_coupler_helpers_register_components_and_add_exchanges() -> None:
    from setups.coupler_helpers import add_exchanges, build_coupler

    grid = make_test_grid(name="shared")
    ocean = DataComponent(name="OCN", grid=grid)
    atmosphere = DataComponent(name="ATM", grid=grid)
    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1)
    run_sequence = RunSequence(order=["OCN", "ATM"])
    exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["sea_surface_temperature"],
        regridder_factory=bilinear,
    )

    coupler = build_coupler(
        clock=clock,
        components=(ocean, atmosphere),
        run_sequence=run_sequence,
    )
    add_exchanges(coupler, (exchange,))

    assert coupler.clock is clock
    assert tuple(coupler.components) == ("OCN", "ATM")
    assert coupler.run_sequence is run_sequence
    assert coupler.exchanges == [exchange]


@pytest.mark.fast_always
def test_multi_exchange_setup_scripts_use_shared_add_exchanges_helper() -> None:
    multi_exchange_scripts = (
        Path("setups/run_data_driver.py"),
        Path("setups/run_jcm_with_verosdata.py"),
        Path("setups/run_jcm_with_veros.py"),
        Path("setups/run_jcm_with_slab.py"),
        Path("setups/run_slab_driver.py"),
    )

    for path in multi_exchange_scripts:
        source = path.read_text(encoding="utf-8")
        assert "add_exchanges" in source, path
        assert "cpl.add_exchange(" not in source, path


@pytest.mark.fast_always
def test_runtime_state_is_separate_from_public_component_objects() -> None:
    assert hasattr(components_module, "DataComponent")
    component = components_module.DataComponent(
        name="ATM",
        grid=make_test_grid(name="api-boundary"),
    )
    runtime_state = RuntimeComponentState(
        data=RuntimeFieldStore.empty(),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
    )

    assert not isinstance(runtime_state, Component)
    assert not hasattr(runtime_state, "name")
    assert not hasattr(runtime_state, "grid")
    assert not hasattr(runtime_state, "settings")
    assert not hasattr(component, "incoming")
    assert not hasattr(component, "outgoing")
    assert not hasattr(component, "with_data")


@pytest.mark.fast_always
def test_examples_import_run_sequence_from_top_level_public_api() -> None:
    for path in Path("setups").glob("run_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from vercor.coupler import RunSequence" not in source
        if "RunSequence" in source:
            public_import_lines = [
                line
                for line in source.splitlines()
                if line.startswith("from vercor import ")
            ]
            assert any("RunSequence" in line for line in public_import_lines), path


@pytest.mark.fast_always
def test_setup_factories_are_primary_concrete_component_api() -> None:
    from setups.slab import (
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

    from setups.data.era5_land import make_era5_land
    from setups.external.veros_gcm import make_veros_gcm

    assert callable(make_era5_land)
    assert callable(make_veros_gcm)


@pytest.mark.fast_always
def test_old_concrete_component_packages_are_removed() -> None:
    assert not Path("vercor/components/slab").exists()
    assert not Path("vercor/components/data").exists()
    assert not Path("vercor/components/external").exists()


@pytest.mark.fast_always
def test_setup_modules_do_not_subclass_component_contracts() -> None:
    forbidden_bases = {"Component", "DataComponent", "HostRuntimeComponent"}
    for path in Path("setups").rglob("*.py"):
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
        Path("setups/external/jax_gcm.py"),
        Path("setups/external/veros_gcm.py"),
        Path("setups/external/camulator.py"),
        Path("setups/data/camulator_land.py"),
    ):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in source, f"{path} borrows {marker}"


@pytest.mark.fast_always
def test_camulator_adapters_share_runtime_cursor_state_transition_helper() -> None:
    for path in (
        Path("setups/external/camulator.py"),
        Path("setups/data/camulator_land.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "CamulatorRuntimeCursor" in source, path
        assert "runtime_forcing_index(" not in source, path
        assert "timestep_counter += 1" not in source, path


@pytest.mark.fast_always
def test_jcm_land_uses_single_coordinate_conversion_helper() -> None:
    source = Path("setups/data/jcm_land.py").read_text(encoding="utf-8")

    assert "def _jcm_coordinates_in_degrees" in source
    assert "def _coordinates_in_degrees" not in source


@pytest.mark.fast_always
def test_bilinear_interpolator_removes_unused_cartesian_helper() -> None:
    source = Path("vercor/interpolators/bilinear_rectilinear.py").read_text(
        encoding="utf-8"
    )

    assert "def _geo_to_cart(" not in source


@pytest.mark.fast_always
def test_jax_gcm_factory_does_not_attach_test_only_setup_state() -> None:
    jax_gcm_source = Path("setups/external/jax_gcm.py").read_text(encoding="utf-8")
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


@pytest.mark.fast_always
def test_data_and_host_factories_return_core_contract_instances() -> None:
    from setups.data.era5_land import make_era5_land
    from setups.external.camulator import make_camulator_gcm

    assert callable(make_era5_land)
    assert callable(make_camulator_gcm)
    assert issubclass(DataComponent, Component)
    assert issubclass(HostRuntimeComponent, Component)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "import_statement",
    (
        "import setups.external.jax_gcm",
        "from setups.external import jax_gcm as jax_gcm_module",
        "from setups.external import make_jax_gcm",
        "import setups.data.era5_land",
        "from setups.data import make_era5_land",
        "import setups.external.camulator",
        "import setups.external.camulator_state",
    ),
)
def test_unrelated_setup_imports_do_not_initialize_optional_adapters(
    import_statement: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", import_statement],
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
