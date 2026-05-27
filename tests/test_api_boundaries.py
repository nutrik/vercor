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
import vercor.components._validation as validation_module
import vercor.components.base as base_module
import vercor.components.factories as factories_module
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
    assert components_module.data_component is factories_module.data_component
    assert (
        components_module.differentiable_component
        is factories_module.differentiable_component
    )
    assert components_module.host_component is factories_module.host_component
    assert validation_module.validate_component_setup is not None
    assert not hasattr(base_module, "validate_component_setup")
    assert not hasattr(components_module, "validate_component_setup")
    assert not hasattr(base_module, "data_component")
    assert not hasattr(base_module, "differentiable_component")
    assert not hasattr(base_module, "host_component")
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
    factories_source = Path("vercor/components/factories.py").read_text(
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
    assert "def _install_lifecycle_hooks(" not in base_source
    assert "def _callable_component_from_model(" not in base_source
    assert "def data_component(" not in base_source
    assert "def differentiable_component(" not in base_source
    assert "def host_component(" not in base_source
    assert "def _install_lifecycle_hooks(" in factories_source
    assert "def _callable_component_from_model(" in factories_source
    assert factories_source.count("_create_callable_component(") == 1
    assert "def data_component(" in factories_source
    assert "def differentiable_component(" in factories_source
    assert "def host_component(" in factories_source
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

    helper_source = Path("vercor/setups/data/_component_helpers.py").read_text(
        encoding="utf-8"
    )
    era5_atmosphere_source = Path("vercor/setups/data/era5_atmosphere.py").read_text(
        encoding="utf-8"
    )

    assert "cast(Any, component).DATA_FILES" not in helper_source
    assert "cast(Any, component).hyai" not in era5_atmosphere_source
    assert "cast(Any, component).hybi" not in era5_atmosphere_source
    assert "cast(Any, component).hyam" not in era5_atmosphere_source
    assert "cast(Any, component).hybm" not in era5_atmosphere_source


@pytest.mark.fast_always
def test_setup_forcing_reader_facade_is_removed() -> None:
    import vercor.forcing_data as forcing_data_module

    assert callable(forcing_data_module.read_forcing)
    assert not Path("vercor/setups/data/forcing.py").exists()


@pytest.mark.fast_always
def test_setup_coupler_helpers_register_components_and_add_exchanges() -> None:
    from vercor.setups.coupler_helpers import add_exchanges, build_coupler

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
        Path("examples/run_data_driver.py"),
        Path("examples/run_jcm_with_verosdata.py"),
        Path("examples/run_jcm_with_veros.py"),
        Path("examples/run_jcm_with_slab.py"),
        Path("examples/run_slab_driver.py"),
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
    for path in Path("examples").glob("run_*.py"):
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
    from vercor.setups.slab import (
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

    from vercor.setups.data.era5_land import make_era5_land
    from vercor.setups.external.veros_gcm import make_veros_gcm

    assert callable(make_era5_land)
    assert callable(make_veros_gcm)


@pytest.mark.fast_always
def test_old_concrete_component_packages_are_removed() -> None:
    assert not Path("vercor/components/slab").exists()
    assert not Path("vercor/components/data").exists()
    assert not Path("vercor/components/external").exists()
    assert not Path("setups").exists()


@pytest.mark.fast_always
def test_setup_modules_do_not_subclass_component_contracts() -> None:
    forbidden_bases = {"Component", "DataComponent", "HostRuntimeComponent"}
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
        Path("vercor/setups/external/jax_gcm.py"),
        Path("vercor/setups/external/veros_gcm.py"),
        Path("vercor/setups/external/camulator.py"),
        Path("vercor/setups/external/camulator_land.py"),
    ):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in source, f"{path} borrows {marker}"


@pytest.mark.fast_always
def test_setup_adapters_do_not_import_runtime_context_or_store_internals() -> None:
    forbidden_markers = (
        "from vercor.runtime.contexts import ComponentInitContext",
        "from vercor.runtime.contexts import RuntimeStepContext",
        "from vercor.runtime import RuntimeFieldStore",
        "RuntimeFieldStore.from_mapping",
    )
    for path in Path("vercor/setups").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in source, f"{path} imports runtime internals"


@pytest.mark.fast_always
def test_shared_helpers_have_core_owners_not_setup_or_regridder_owners() -> None:
    import vercor.calendar as calendar_module
    import vercor.clock as clock_module
    import vercor.field_names as field_names_module
    import vercor.fluxes.vertical_coordinates as vertical_module
    import vercor.grid_geometry as grid_geometry_module
    import vercor.grid_masks as grid_masks_module
    import vercor.pytree_utils as pytree_utils_module
    import vercor.physical_constants as physical_constants_module
    import vercor.exchange as exchange_module

    assert callable(calendar_module.is_leap_year)
    assert callable(calendar_module.daily_forcing_index)
    assert not hasattr(clock_module, "DateTime360")
    assert not hasattr(clock_module, "DateTime365")
    assert not hasattr(clock_module, "ModelDateTime")
    assert callable(grid_geometry_module.make_rectilinear_grid)
    assert callable(grid_geometry_module.centers_to_edges)
    assert not hasattr(grid_masks_module, "grids_identical")
    assert callable(vertical_module.compute_pressure_levels)
    assert callable(vertical_module.compute_sigma_pressure_levels)
    assert callable(vertical_module.compute_hybrid_pressure_levels)
    assert callable(vertical_module.compute_hybrid_sigma_full_level_altitudes)
    assert callable(vertical_module.get_altitudes_sigma_levels)
    assert callable(pytree_utils_module.asfloat)
    assert callable(pytree_utils_module.mean_leaf)
    assert callable(pytree_utils_module.stack_objects)
    assert callable(pytree_utils_module.unwrap_leading_dims)
    assert "gravity" in physical_constants_module.PHYSICAL_CONSTANT_SETTINGS
    assert not hasattr(exchange_module, "VALID_EXCHANGE_FIELD_NAMES")
    assert "sea_surface_temperature" in field_names_module.VALID_EXCHANGE_FIELD_NAMES

    clock_source = Path("vercor/clock.py").read_text(encoding="utf-8")
    runtime_time_source = Path("vercor/runtime/time.py").read_text(encoding="utf-8")
    runtime_validation_source = Path("vercor/runtime/validation.py").read_text(
        encoding="utf-8"
    )
    exchange_source = Path("vercor/exchange.py").read_text(encoding="utf-8")
    regridder_base_source = Path("vercor/regridders/base.py").read_text(
        encoding="utf-8"
    )
    settings_source = Path("vercor/settings.py").read_text(encoding="utf-8")
    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")
    regridder_init = Path("vercor/regridders/__init__.py").read_text(encoding="utf-8")
    grid_masks_source = Path("vercor/grid_masks.py").read_text(encoding="utf-8")
    topology_source = Path("vercor/runtime/topology.py").read_text(encoding="utf-8")
    jax_gcm_tools_source = Path("vercor/setups/external/jax_gcm_tools.py").read_text(
        encoding="utf-8"
    )

    assert "class _ModelDateTimeBase" not in clock_source
    assert "class DateTime365" not in clock_source
    assert "class DateTime360" not in clock_source
    assert "def runtime_daily_index(" not in runtime_time_source
    assert "from vercor.exchange import VALID_EXCHANGE_FIELD_NAMES" not in (
        runtime_validation_source
    )
    assert "from vercor.field_names import VALID_EXCHANGE_FIELD_NAMES" in (
        runtime_validation_source
    )
    assert "from vercor.field_names import" not in exchange_source
    assert (
        "VALID_EXCHANGE_FIELD_NAMES as VALID_EXCHANGE_FIELD_NAMES"
        not in exchange_source
    )
    assert "VALID_EXCHANGE_FIELD_NAMES: list[str]" not in exchange_source
    assert "def _compute_has_identical_grids(" not in regridder_base_source
    assert "grids_identical(" in regridder_base_source
    assert "BilinearRectilinearInterpolator" not in regridder_base_source
    assert "ConservativeRectilinearRemapper" not in regridder_base_source
    assert "class SupportsScalarVectorInterpolation" in regridder_base_source
    assert '"gravity": Settings(' not in settings_source
    assert "PHYSICAL_CONSTANT_SETTINGS" in settings_source
    assert "Incorrect component name" not in coupler_source
    assert "def validate_component_topology_names(" in topology_source
    assert "make_rectilinear_grid" not in regridder_init
    assert "centers_to_edges" not in regridder_init
    assert "compute_land_mask" not in regridder_init
    assert "def compute_land_mask(" in grid_masks_source
    assert "def get_component(" not in grid_masks_source
    assert "def get_component(" in topology_source
    assert "def compute_pressure_levels(" not in jax_gcm_tools_source
    assert "def get_altitudes_sigma_levels(" not in jax_gcm_tools_source
    assert "def mean_leaf(" not in jax_gcm_tools_source
    assert "def stack_objects(" not in jax_gcm_tools_source
    assert "def unwrap_leading_dims(" not in jax_gcm_tools_source


@pytest.mark.fast_always
def test_setup_helper_and_external_output_ownership_boundaries() -> None:
    import vercor.diagnostics as diagnostics_module
    import vercor.host_arrays as host_arrays_module
    import vercor.setups.external.camulator as camulator_module
    import vercor.setups.external.camulator_fields as camulator_fields_module
    import vercor.setups.external.camulator_land as camulator_land_module
    import vercor.setups.external.camulator_runtime_settings as camulator_runtime_settings_module
    import vercor.setups.external.jax_gcm as jax_gcm_module
    import vercor.setups.external.veros_fluxes as veros_fluxes_module
    import vercor.setups.external.veros_gcm as veros_gcm_module
    import vercor.setups.external.veros_setup as veros_setup_module
    import vercor.setups.external.veros_state as veros_state_module

    assert callable(host_arrays_module.transposed_host_array)
    assert callable(diagnostics_module.component_vector_speed)
    assert callable(camulator_land_module.make_camulator_land)
    assert callable(camulator_fields_module._prepare_camulator_surface_forcing)
    assert callable(camulator_runtime_settings_module.configure_camulator_runtime)
    assert callable(veros_fluxes_module.compute_fluxes)
    assert callable(veros_state_module.copy_state)
    assert hasattr(veros_setup_module, "CustomGlobalFourDegree")
    assert not hasattr(jax_gcm_module, "_map_jcm_output_fields")
    assert not hasattr(jax_gcm_module, "_prepare_surface_temperature_forcing")
    assert not hasattr(camulator_module, "_map_camulator_prediction_arrays")
    assert not hasattr(camulator_module, "_prepare_camulator_surface_forcing")
    assert not hasattr(veros_gcm_module, "compute_fluxes")
    assert not hasattr(veros_gcm_module, "copy_state")
    assert not hasattr(veros_gcm_module, "set_variable")
    jax_gcm_source = Path("vercor/setups/external/jax_gcm.py").read_text(
        encoding="utf-8"
    )
    jax_gcm_fields_source = Path("vercor/setups/external/jax_gcm_fields.py").read_text(
        encoding="utf-8"
    )
    camulator_source = Path("vercor/setups/external/camulator.py").read_text(
        encoding="utf-8"
    )
    camulator_fields_source = Path(
        "vercor/setups/external/camulator_fields.py"
    ).read_text(encoding="utf-8")
    camulator_tensors_source = Path(
        "vercor/setups/external/camulator_tensors.py"
    ).read_text(encoding="utf-8")
    camulator_init_source = Path("vercor/setups/external/camulator_init.py").read_text(
        encoding="utf-8"
    )
    camulator_runtime_settings_source = Path(
        "vercor/setups/external/camulator_runtime_settings.py"
    ).read_text(encoding="utf-8")
    veros_gcm_source = Path("vercor/setups/external/veros_gcm.py").read_text(
        encoding="utf-8"
    )
    camulator_imports_source = Path(
        "vercor/setups/external/camulator_imports.py"
    ).read_text(encoding="utf-8")

    assert Path("vercor/setups/external/camulator_land.py").exists()
    assert Path("vercor/setups/external/jax_gcm_output.py").exists()
    assert Path("vercor/setups/external/jax_gcm_fields.py").exists()
    assert Path("vercor/setups/external/camulator_output.py").exists()
    assert Path("vercor/setups/external/camulator_fields.py").exists()
    assert Path("vercor/setups/external/camulator_runtime_settings.py").exists()
    assert Path("vercor/setups/external/camulator_wind_filter.py").exists()
    assert Path("vercor/setups/external/veros_fluxes.py").exists()
    assert Path("vercor/setups/external/veros_setup.py").exists()
    assert Path("vercor/setups/external/veros_state.py").exists()
    assert not Path("vercor/setups/jax_array_helpers.py").exists()
    assert not Path("vercor/setups/data/camulator_land.py").exists()
    assert not Path("vercor/setups/external/windpp.py").exists()
    assert "from vercor.runtime.validation import" not in jax_gcm_source
    assert "def asfloat(" not in jax_gcm_source
    assert "def _cleanup_surface_temperature_fields(" not in jax_gcm_source
    assert "def _prepare_surface_temperature_forcing(" not in jax_gcm_source
    assert "def _map_jcm_output_fields(" not in jax_gcm_source
    assert "def _cleanup_surface_temperature_fields(" in jax_gcm_fields_source
    assert "def _should_write_output(" not in jax_gcm_source
    assert "def _write_output(" not in jax_gcm_source
    assert "os.environ[" not in camulator_source
    assert "def configure_camulator_runtime(" in camulator_runtime_settings_source
    assert "def _prepare_camulator_surface_forcing(" not in camulator_source
    assert "def _map_camulator_prediction_arrays(" not in camulator_source
    assert "def _prepare_camulator_surface_forcing(" in camulator_fields_source
    assert "def _map_camulator_prediction_arrays(" in camulator_fields_source
    assert "def _torch_tensor_from_jax_array(" not in camulator_source
    assert "def _torch_tensor_from_jax_array(" in camulator_tensors_source
    assert "def add_init_noise(" not in camulator_source
    assert "def add_init_noise(" in camulator_init_source
    assert "def _credit_output_functions(" not in camulator_source
    assert "def _write_camulator_prediction_output(" not in camulator_source
    assert "class CustomGlobalFourDegree" not in veros_gcm_source
    assert "def compute_fluxes(" not in veros_gcm_source
    assert "def copy_state(" not in veros_gcm_source
    assert "def set_variable(" not in veros_gcm_source
    assert "from vercor.setups.external.camulator_wind_filter import" in (
        camulator_imports_source
    )


@pytest.mark.fast_always
def test_common_exchange_recipes_are_centralized_for_examples() -> None:
    import vercor.setups.exchange_recipes as exchange_recipes_module

    required_recipes = (
        "ATMOSPHERE_TO_DATA_OCEAN_FIELDS",
        "ATMOSPHERE_TO_LAND_RADIATION_FIELDS",
        "ATMOSPHERE_TO_LAND_STATE_FIELDS",
        "LAND_TO_ATMOSPHERE_SURFACE_FIELDS",
        "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS",
        "SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS",
        "SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS",
    )
    for recipe_name in required_recipes:
        assert hasattr(exchange_recipes_module, recipe_name)

    recipe_users = (
        Path("examples/run_jcm_with_verosdata.py"),
        Path("examples/run_jcm_with_era5data.py"),
        Path("examples/run_jcm_with_slab.py"),
        Path("examples/run_camulator_with_veros.py"),
        Path("examples/run_data_driver.py"),
        Path("examples/run_veros_with_era5data.py"),
        Path("examples/profile_runtime.py"),
    )
    for path in recipe_users:
        source = path.read_text(encoding="utf-8")
        assert "from vercor.setups.exchange_recipes import" in source, path


@pytest.mark.fast_always
def test_assets_and_diagnostics_have_focused_ownership_boundaries() -> None:
    import vercor.assets as assets_module
    import vercor.diagnostics as diagnostics_module
    import vercor.setups.data.assets as setup_assets_module

    assert hasattr(assets_module, "ensure_registered_asset")
    assert not hasattr(assets_module, "get_forcing_data")
    assert not hasattr(assets_module, "_FORCING_ASSETS")
    assert hasattr(setup_assets_module, "get_forcing_data")

    assert diagnostics_module.combine_surface_temperatures is not None
    assert diagnostics_module.print_component_field_means_table is not None
    assert diagnostics_module.plot_component_scalar_vector_comparison is not None
    assert Path("vercor/diagnostics/fields.py").exists()
    assert Path("vercor/diagnostics/tables.py").exists()
    assert Path("vercor/diagnostics/plotting.py").exists()
    assert not Path("vercor/diagnostics.py").exists()


@pytest.mark.fast_always
def test_camulator_adapters_share_runtime_cursor_state_transition_helper() -> None:
    for path in (
        Path("vercor/setups/external/camulator.py"),
        Path("vercor/setups/external/camulator_land.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "CamulatorRuntimeCursor" in source, path
        assert "runtime_forcing_index(" not in source, path
        assert "timestep_counter += 1" not in source, path


@pytest.mark.fast_always
def test_camulator_state_facade_is_removed() -> None:
    focused_modules = (
        Path("vercor/setups/external/camulator_imports.py"),
        Path("vercor/setups/external/camulator_forcing.py"),
        Path("vercor/setups/external/camulator_tensors.py"),
        Path("vercor/setups/external/camulator_stepper.py"),
        Path("vercor/setups/external/camulator_init.py"),
    )
    for path in focused_modules:
        assert path.exists(), path

    assert not Path("vercor/setups/external/camulator_state.py").exists()


@pytest.mark.fast_always
def test_jcm_land_uses_single_coordinate_conversion_helper() -> None:
    source = Path("vercor/setups/data/jcm_land.py").read_text(encoding="utf-8")

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
    jax_gcm_source = Path("vercor/setups/external/jax_gcm.py").read_text(
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

    jcm_slab_source = Path("examples/run_jcm_with_slab.py").read_text(encoding="utf-8")
    assert 'getattr(atm, "model")' not in jcm_slab_source


@pytest.mark.fast_always
def test_data_and_host_factories_return_core_contract_instances() -> None:
    from vercor.setups.data.era5_land import make_era5_land
    from vercor.setups.external.camulator import make_camulator_gcm

    assert callable(make_era5_land)
    assert callable(make_camulator_gcm)
    assert issubclass(DataComponent, Component)
    assert issubclass(HostRuntimeComponent, Component)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "import_statement",
    (
        "import vercor.setups.external.jax_gcm",
        "from vercor.setups.external import jax_gcm as jax_gcm_module",
        "from vercor.setups.external import make_jax_gcm",
        "import vercor.setups.data.era5_land",
        "from vercor.setups.data import make_era5_land",
        "import vercor.setups.external.camulator",
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
