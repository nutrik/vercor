from __future__ import annotations

from datetime import datetime
from inspect import signature
from pathlib import Path
import ast
import importlib
import subprocess
import sys

import pytest

import vercor
import vercor.components as components_module
import vercor.components.base as base_module
import vercor.components.contexts as component_contexts_module
import vercor.components.contracts as component_contracts_module
import vercor.components.data as data_module
import vercor.components.host as host_module
import vercor.components.setup_validation as setup_validation_module
from tests._architecture_support import package_import_cycles
from tests._coverage_support import make_test_grid
from vercor.components.base import Component
from vercor.components.data import DataComponent
from vercor.components.host import HostRuntimeComponent
from vercor.clock import Clock
from vercor.coupler import Coupler
from vercor.exchange import Exchange
from vercor.runtime.state import RuntimeComponentState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.regridders import bilinear


@pytest.mark.fast_always
def test_top_level_exports_public_orchestration_and_component_author_api() -> None:
    expected_public_names = {
        "Clock",
        "Component",
        "ComponentCreatePayloadHook",
        "ComponentFieldSpec",
        "ComponentInitializeHook",
        "ComponentPrefillHook",
        "ComponentSetupContext",
        "ComponentStepContext",
        "ComponentStepResult",
        "ComponentValidateHook",
        "Coupler",
        "DataComponent",
        "Exchange",
        "HostRuntimeComponent",
        "RectilinearGrid",
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
    removed_compatibility_names = {
        "RunSequence",
        "data_component",
        "differentiable_component",
        "host_component",
        "make_data_component",
        "make_differentiable_component",
        "make_host_component",
    }

    assert expected_public_names.issubset(set(vercor.__all__))
    assert runtime_internal_names.isdisjoint(set(vercor.__all__))
    assert removed_compatibility_names.isdisjoint(set(vercor.__all__))

    assert vercor.Component is Component
    assert (
        vercor.ComponentCreatePayloadHook
        is component_contracts_module.ComponentCreatePayloadHook
    )
    assert vercor.ComponentFieldSpec is component_contracts_module.ComponentFieldSpec
    assert (
        vercor.ComponentInitializeHook
        is component_contracts_module.ComponentInitializeHook
    )
    assert (
        vercor.ComponentPrefillHook is component_contracts_module.ComponentPrefillHook
    )
    assert (
        vercor.ComponentSetupContext is component_contexts_module.ComponentSetupContext
    )
    assert vercor.ComponentStepContext is component_contexts_module.ComponentStepContext
    assert vercor.ComponentStepResult is component_contracts_module.ComponentStepResult
    assert (
        vercor.ComponentValidateHook is component_contracts_module.ComponentValidateHook
    )
    data_component_type = getattr(components_module, "DataComponent", None)
    assert data_component_type is not None
    assert getattr(vercor, "DataComponent", None) is data_component_type
    assert vercor.HostRuntimeComponent is HostRuntimeComponent
    for name in (*runtime_internal_names, *removed_compatibility_names):
        assert not hasattr(vercor, name)


@pytest.mark.fast_always
def test_removed_compatibility_modules_are_not_importable() -> None:
    removed_modules = (
        "vercor.components.factories",
        "vercor.run_sequence",
    )

    for module_name in removed_modules:
        with pytest.raises(ModuleNotFoundError, match=module_name):
            importlib.import_module(module_name)


@pytest.mark.fast_always
def test_components_package_exports_only_component_author_contracts() -> None:
    contracts_module = importlib.import_module("vercor.components.contracts")
    private_contracts_module = importlib.import_module("vercor.components._contracts")
    imported_data_module = importlib.import_module("vercor.components.data")
    imported_host_module = importlib.import_module("vercor.components.host")

    assert base_module.__all__ == ["Component"]
    assert not hasattr(base_module, "ComponentFieldSpec")
    assert not hasattr(base_module, "ComponentSetupContext")
    assert not hasattr(base_module, "ComponentStepContext")
    assert not hasattr(base_module, "ComponentStepResult")
    assert not hasattr(base_module, "DataComponent")
    assert not hasattr(base_module, "HostRuntimeComponent")

    assert components_module.__all__ == [
        "Component",
        "ComponentCreatePayloadHook",
        "ComponentFieldSpec",
        "ComponentInitializeHook",
        "ComponentPrefillHook",
        "ComponentSetupContext",
        "ComponentStepContext",
        "ComponentStepResult",
        "ComponentValidateHook",
        "DataComponent",
        "HostRuntimeComponent",
    ]
    assert components_module.Component is Component
    assert (
        components_module.ComponentCreatePayloadHook
        is contracts_module.ComponentCreatePayloadHook
    )
    assert components_module.ComponentFieldSpec is contracts_module.ComponentFieldSpec
    assert (
        components_module.ComponentInitializeHook
        is contracts_module.ComponentInitializeHook
    )
    assert (
        components_module.ComponentPrefillHook is contracts_module.ComponentPrefillHook
    )
    assert (
        components_module.ComponentSetupContext
        is component_contexts_module.ComponentSetupContext
    )
    assert (
        components_module.ComponentStepContext
        is component_contexts_module.ComponentStepContext
    )
    assert components_module.ComponentStepResult is contracts_module.ComponentStepResult
    assert (
        components_module.ComponentValidateHook
        is contracts_module.ComponentValidateHook
    )
    assert data_module is imported_data_module
    assert host_module is imported_host_module
    assert components_module.DataComponent is data_module.DataComponent
    assert components_module.HostRuntimeComponent is host_module.HostRuntimeComponent
    assert setup_validation_module.validate_component_setup is not None
    assert "FieldDefaults" not in contracts_module.__all__
    assert "FieldDefaults" not in private_contracts_module.__all__
    assert not hasattr(contracts_module, "FieldDefaults")
    assert not hasattr(private_contracts_module, "FieldDefaults")
    assert not hasattr(base_module, "validate_component_setup")
    assert not hasattr(components_module, "validate_component_setup")
    assert not hasattr(base_module, "data_component")
    assert not hasattr(base_module, "differentiable_component")
    assert not hasattr(base_module, "host_component")
    assert not hasattr(components_module, "data_component")
    assert not hasattr(components_module, "differentiable_component")
    assert not hasattr(components_module, "host_component")
    assert not hasattr(components_module, "make_data_component")
    assert not hasattr(components_module, "make_differentiable_component")
    assert not hasattr(components_module, "make_host_component")
    assert not hasattr(components_module, "RuntimeComponentState")
    assert not hasattr(components_module, "ComponentInitContext")
    assert not hasattr(components_module, "RuntimeStepContext")


@pytest.mark.fast_always
def test_runtime_setup_validation_uses_single_component_entrypoint() -> None:
    removed_wrapper_name = "validate_" + "registered_component_setup"
    offenders = [
        str(path)
        for path in sorted(Path("vercor").glob("**/*.py"))
        if removed_wrapper_name in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


@pytest.mark.fast_always
def test_setup_and_examples_do_not_import_removed_component_factories() -> None:
    removed_names = {
        "data_component",
        "differentiable_component",
        "host_component",
    }
    import_modules = {
        "vercor",
        "vercor.components",
    }
    scanned_paths = (
        *Path("vercor/setups").glob("**/*.py"),
        *Path("examples").glob("**/*.py"),
    )
    offenders: list[str] = []

    for path in sorted(scanned_paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.ImportFrom)
                or node.module not in import_modules
            ):
                continue
            imported_removed_names = removed_names.intersection(
                alias.name for alias in node.names
            )
            for name in sorted(imported_removed_names):
                offenders.append(f"{path}:{node.lineno}:{name}")

    assert offenders == []


@pytest.mark.fast_always
def test_obsolete_compatibility_api_surfaces_are_removed() -> None:
    import vercor.forcing_data as forcing_data_module
    import vercor.runtime as runtime_module
    import vercor.settings as settings_module
    import vercor.setups.external as external_module
    import vercor.setups.external.camulator_tensors as camulator_tensors_module
    import vercor.setups.external.jax_gcm as jax_gcm_module

    removed_runtime_reexports = {
        "RuntimeComponentContract",
        "RuntimeComponentState",
        "RuntimeCouplerState",
        "RuntimeFieldStore",
        "RuntimeStepInfo",
        "append_unique_runtime_fields",
        "build_runtime_contracts",
        "dispatch_component_exchanges",
        "exchange_key_name",
        "flatten_exchange_fields",
    }
    for name in removed_runtime_reexports:
        assert not hasattr(runtime_module, name)
    assert getattr(runtime_module, "__all__", []) == []
    assert not Path("vercor/runtime/contexts.py").exists()

    assert not hasattr(jax_gcm_module, "JAXGCMRuntimePayload")
    assert "JAXGCMRuntimePayload" not in external_module.__all__
    assert "JAXGCMRuntimePayload" not in external_module._LAZY_EXPORTS
    assert not hasattr(jax_gcm_module, "JCMState")
    assert "JCMState" not in getattr(jax_gcm_module, "__all__", [])
    assert "JCMState" not in external_module.__all__
    assert "JCMState" not in external_module._LAZY_EXPORTS

    assert not hasattr(settings_module, "ComponentSettings")
    assert not hasattr(forcing_data_module, "ComponentForcingData")
    assert not hasattr(camulator_tensors_module.TensorVariableIndex, "to_mapping")
    assert not hasattr(camulator_tensors_module.StateVariableAccessor, "get_var_info")
    assert not hasattr(
        camulator_tensors_module.StateVariableAccessor,
        "list_available_vars",
    )


@pytest.mark.fast_always
def test_active_progress_does_not_advertise_removed_compatibility_surfaces() -> None:
    progress_source = Path("PROGRESS.md").read_text(encoding="utf-8")

    stale_progress_markers = (
        "while preserving the\n  existing private compatibility aliases for tests and profiling helpers",
        "preserving public and private compatibility methods used by tests",
        "keeping `StateVariableAccessor.get_var_info(...)` dictionary-compatible",
        "compatibility `JAXGCMRuntimePayload` reexport",
        "preserving stable package aggregators, `ComponentSettings`",
        "preserving `vercor.clock` compatibility reexports",
        "leaving old flux utility import paths as\n  compatibility aliases",
        "a thin compatibility facade",
        "Preserved intentional compatibility surfaces, including settings attribute\n"
        "  compatibility, `ComponentSettings`",
        "reexports, `ComponentForcingData._read_forcing()`",
        "`Coupler._run_scanned_runtime()`",
        "`_runtime_state_from_components()`",
    )

    for marker in stale_progress_markers:
        assert marker not in progress_source


@pytest.mark.fast_always
def test_coupler_private_compatibility_aliases_are_removed() -> None:
    removed_names = (
        "_regridders",
        "_binary_masks",
        "_fractional_masks",
        "_runtime_contracts",
        "_compiled_runtime_cache",
        "_runtime_interrupts",
        "_runtime_state_from_components",
        "_validate_runtime_state",
        "_prepare_runtime_state",
        "_runtime_dispatch_context",
        "_runtime_run_context",
        "_run_scanned_runtime",
    )
    for name in removed_names:
        assert not hasattr(Coupler, name)


@pytest.mark.fast_always
def test_callable_author_api_does_not_expose_legacy_field_seed_keyword() -> None:
    public_callables = (
        components_module.Component.from_model,
        components_module.HostRuntimeComponent.from_model,
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
    public_contracts_source = Path("vercor/components/contracts.py").read_text(
        encoding="utf-8"
    )
    data_source = Path("vercor/components/data.py").read_text(encoding="utf-8")
    host_source = Path("vercor/components/host.py").read_text(encoding="utf-8")
    callable_source = Path("vercor/components/_callable_wrappers.py").read_text(
        encoding="utf-8"
    )
    runtime_fields_source = Path("vercor/components/_runtime_fields.py").read_text(
        encoding="utf-8"
    )
    runtime_validation_source = Path(
        "vercor/components/_runtime_validation.py"
    ).read_text(encoding="utf-8")
    core_runtime_validation_source = Path("vercor/runtime/validation.py").read_text(
        encoding="utf-8"
    )
    runtime_execution_source = Path("vercor/components/runtime_execution.py").read_text(
        encoding="utf-8"
    )
    lifecycle_source = Path("vercor/components/_lifecycle.py").read_text(
        encoding="utf-8"
    )
    validation_source = Path("vercor/components/setup_validation.py").read_text(
        encoding="utf-8"
    )

    assert "class ComponentFieldSpec" in public_contracts_source
    assert "class ComponentStepResult" in public_contracts_source
    assert "ComponentInitializeHook =" in public_contracts_source
    assert "ComponentCreatePayloadHook =" in public_contracts_source
    assert "ComponentPrefillHook =" in public_contracts_source
    assert "ComponentValidateHook =" in public_contracts_source
    assert "class ComponentFieldSpec" not in contracts_source
    assert "class ComponentStepResult" not in contracts_source
    assert "class DataComponent" in data_source
    assert "class HostRuntimeComponent" in host_source
    assert "class DataComponent" not in base_source
    assert "class HostRuntimeComponent" not in base_source
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
    assert "def require_runtime_fields(" not in runtime_fields_source
    assert "def validate_declared_runtime_fields(" not in runtime_fields_source
    assert "def require_runtime_fields(" in runtime_validation_source
    assert "def validate_declared_runtime_fields(" in runtime_validation_source
    assert (
        "def validate_runtime_component_data_field("
        not in core_runtime_validation_source
    )
    assert "def component_requires_host_runtime(" in runtime_execution_source
    assert "def host_component_names(" in runtime_execution_source
    assert "def step_component_runtime_state(" in runtime_execution_source
    assert "from vercor.components._protocols import" in runtime_execution_source
    assert "ComponentExecutionProtocol" not in runtime_execution_source
    assert "HostRuntimeExecutionProtocol" in runtime_execution_source
    assert "if TYPE_CHECKING:" in runtime_execution_source
    assert "from vercor.components.base import Component" in runtime_execution_source
    assert "from vercor.components.host import HostRuntimeComponent" not in (
        runtime_execution_source
    )
    assert "isinstance(component, HostRuntimeExecutionProtocol)" in (
        runtime_execution_source
    )
    assert "isinstance(component, HostRuntimeComponent)" not in (
        runtime_execution_source
    )
    assert "def validate_component_setup" in validation_source
    assert "def _author_field_spec(" not in base_source
    assert "def component_field_spec(" not in contracts_source
    assert "def _install_lifecycle_hooks(" not in base_source
    assert "def _install_lifecycle_hooks(" not in callable_source
    assert "def _callable_component_from_model(" not in base_source
    assert "from vercor.components.factories import" not in base_source
    assert "from vercor.components.factories import" not in host_source
    assert not Path("vercor/components/factories.py").exists()
    assert "def data_component(" not in base_source
    assert "def differentiable_component(" not in base_source
    assert "def host_component(" not in base_source
    assert "def install_lifecycle_hooks(" in lifecycle_source
    assert "class ComponentLifecycleHooks" in lifecycle_source
    assert "ComponentInitializeHook =" not in lifecycle_source
    assert "ComponentCreatePayloadHook =" not in lifecycle_source
    assert "ComponentPrefillHook =" not in lifecycle_source
    assert "ComponentValidateHook =" not in lifecycle_source
    assert "from vercor.components import _runtime_fields" not in base_source
    assert "from vercor.components.factories import _install_lifecycle_hooks" not in (
        callable_source
    )
    assert "if TYPE_CHECKING:" in callable_source
    assert "from vercor.components.base import Component" in callable_source
    assert "from vercor.components.host import HostRuntimeComponent" not in (
        callable_source
    )
    assert "class _CallableComponentDefinition" not in callable_source
    assert "def _callable_component_definition(" not in callable_source
    assert "class _CallableComponent" in base_source
    assert "class _CallableHostRuntimeComponent" in host_source
    assert "_required_fields" not in callable_source
    assert "_prefill_fields" not in callable_source
    assert "_field_defaults" not in callable_source
    assert "required_fields:" not in callable_source
    assert "prefill_fields:" not in callable_source
    assert "field_defaults:" not in callable_source
    assert "def apply_callable_step_result" not in callable_source
    assert "def make_callable_component" not in callable_source
    assert "def make_callable_host_component" not in callable_source
    assert "def _create_callable_component" not in callable_source

    private_markers = (
        "class _CallableRuntimeMixin",
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
        "from vercor.runtime.validation import",
        "_initialize_hook",
        "_create_runtime_payload_hook",
    )
    for marker in private_markers:
        assert marker not in base_source

    for marker in (
        "validate_runtime_component_data_field",
        "from vercor.runtime.validation import",
    ):
        assert marker not in runtime_fields_source

    assert "_contracts" not in components_module.__all__
    assert "_callable_wrappers" not in components_module.__all__
    assert "_runtime_fields" not in components_module.__all__
    assert "_runtime_validation" not in components_module.__all__
    assert "runtime_execution" not in components_module.__all__
    assert "setup_validation" not in components_module.__all__
    assert not Path("vercor/components/_runtime_execution.py").exists()
    assert not Path("vercor/components/_validation.py").exists()


@pytest.mark.fast_always
def test_component_base_owns_runtime_access_methods_directly() -> None:
    expected_author_methods = {
        "declare_fields",
        "update_settings",
        "grid_field_defaults",
        "seed_field",
        "seed_fields",
        "seed_declared_defaults",
    }
    expected_runtime_methods = {
        "runtime_fields",
        "runtime_field",
        "has_runtime_field",
        "runtime_field_or",
        "runtime_field_or_zeros_like",
        "with_runtime_fields",
        "apply_step_result",
        "require_runtime_fields",
        "prefill_runtime_fields",
    }
    expected_lifecycle_methods = {
        "initialize",
        "create_runtime_payload",
        "prefill_runtime_state_fields",
        "validate_runtime_state",
    }

    for method_name in (
        expected_author_methods | expected_runtime_methods | expected_lifecycle_methods
    ):
        assert hasattr(Component, method_name)
    assert not hasattr(Component, "seed_zero_field")
    assert not hasattr(Component, "seed_zero_fields")
    assert not hasattr(Component, "seed_constant_field")

    source = Path("vercor/components/base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    component_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Component"
    )
    directly_defined_methods = {
        node.name for node in component_class.body if isinstance(node, ast.FunctionDef)
    }

    assert expected_author_methods.isdisjoint(directly_defined_methods)
    assert expected_runtime_methods.issubset(directly_defined_methods)
    assert expected_lifecycle_methods.isdisjoint(directly_defined_methods)
    assert "ComponentFieldAuthoringMixin" in source
    assert "ComponentRuntimeAccessMixin" not in source
    assert not Path("vercor/components/_runtime_access.py").exists()
    assert "ComponentLifecycleMixin" in source


@pytest.mark.fast_always
def test_component_contract_modules_share_field_name_deduplication_owner() -> None:
    field_names_module = importlib.import_module("vercor.components._field_names")
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
    assert "vercor.components._field_names" in contracts_source
    assert "vercor.components._field_names" in private_contracts_source


@pytest.mark.fast_always
def test_runtime_component_type_imports_are_annotation_only() -> None:
    """Runtime facade modules should not import Component for annotations only."""

    modules_with_annotation_only_component_usage = (
        Path("vercor/coupler.py"),
        Path("vercor/runtime/initialization.py"),
        Path("vercor/runtime/topology.py"),
        Path("vercor/runtime/coupler_state.py"),
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
    from vercor.setups.coupler_helpers import (
        ExchangeSpec,
        add_exchange_specs,
        add_exchanges,
        build_coupler,
        build_exchanges,
    )

    grid = make_test_grid(name="shared")
    ocean = DataComponent(name="OCN", grid=grid)
    atmosphere = DataComponent(name="ATM", grid=grid)
    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1)
    run_sequence = ("OCN", "ATM")
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
    assert coupler.run_sequence == run_sequence
    assert coupler.exchanges == [exchange]

    specs = (
        ExchangeSpec(
            source="OCN",
            destination="ATM",
            field_names=("sea_surface_temperature",),
            regridder_factory=bilinear,
        ),
    )
    built = build_exchanges(specs)
    assert len(built) == 1
    assert built[0].source == exchange.source
    assert built[0].destination == exchange.destination
    assert tuple(built[0].field_names) == tuple(exchange.field_names)
    assert built[0].regridder_factory is exchange.regridder_factory

    second_coupler = build_coupler(
        clock=clock,
        components=(ocean, atmosphere),
        run_sequence=run_sequence,
    )
    add_exchange_specs(second_coupler, specs)
    assert len(second_coupler.exchanges) == 1
    assert second_coupler.exchanges[0].field_names == ("sea_surface_temperature",)


@pytest.mark.fast_always
def test_coupler_run_sequence_is_explicit_empty_schedule_by_default() -> None:
    coupler = vercor.Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1)
    )

    assert coupler.run_sequence == ()

    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")
    assert 'hasattr(self, "run_sequence")' not in coupler_source
    assert 'getattr(self, "run_sequence"' not in coupler_source


@pytest.mark.fast_always
def test_coupler_accepts_plain_component_name_sequences() -> None:
    import numpy as np

    from tests._coverage_support import make_test_grid as _make_grid
    from vercor.setups.coupler_helpers import build_coupler

    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1)
    grid = _make_grid("grid")
    ocean = DataComponent.from_fields(
        "OCN",
        grid,
        fields={"sea_surface_temperature": np.zeros(grid.shape)},
    )
    atmosphere = DataComponent.from_fields(
        "ATM",
        grid,
        fields={"sea_surface_temperature": np.zeros(grid.shape)},
    )

    coupler = build_coupler(
        clock=clock,
        components=(ocean, atmosphere),
        run_sequence=["OCN", "ATM"],
    )

    assert coupler.run_sequence == ("OCN", "ATM")

    coupler.set_components_run_sequence(("ATM", "OCN"))
    assert coupler.run_sequence == ("ATM", "OCN")


@pytest.mark.fast_always
def test_coupler_rejects_string_run_sequence() -> None:
    with pytest.raises(
        TypeError,
        match="run_sequence must be a sequence of component names, not str",
    ):
        Coupler(
            clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            run_sequence="ATM",
        )


@pytest.mark.fast_always
def test_setup_state_reads_run_sequence_as_plain_sequence() -> None:
    setup_state_paths = (
        Path("vercor/setups/external/jax_gcm_state.py"),
        Path("vercor/setups/external/veros_gcm_state.py"),
    )

    for path in setup_state_paths:
        assert ".run_sequence.order" not in path.read_text(encoding="utf-8")


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
        assert "add_exchange_specs" in source, path
        assert "cpl.add_exchange(" not in source, path


@pytest.mark.fast_always
def test_slab_driver_uses_runtime_views_for_ice_diagnostics() -> None:
    slab_source = Path("examples/run_slab_driver.py").read_text(encoding="utf-8")

    assert 'names=("ATM", "OCN", "LND", "ICE")' in slab_source
    assert 'views["ICE"].field("ice_fraction")' in slab_source
    assert 'get_component_state("ICE").data.get("ice_fraction")' not in slab_source


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
def test_examples_do_not_import_removed_run_sequence_api() -> None:
    for path in Path("examples").glob("run_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from vercor.coupler import RunSequence" not in source
        assert "from vercor import RunSequence" not in source
        assert "RunSequence" not in source


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
        "vercor.runtime.contexts",
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
    import vercor.forcing_index as forcing_index_module
    import vercor.fluxes.vertical_coordinates as vertical_module
    import vercor.grid_geometry as grid_geometry_module
    import vercor.grid_masks as grid_masks_module
    import vercor.pytree_utils as pytree_utils_module
    import vercor.physical_constants as physical_constants_module
    import vercor.exchange as exchange_module

    assert callable(calendar_module.is_leap_year)
    removed_calendar_delegates = (
        "gregorian_month_lengths",
        "day_of_year_360_to_gregorian",
        "noleap_day_of_year",
        "daily_forcing_day_of_year",
        "daily_forcing_index",
    )
    for name in removed_calendar_delegates:
        assert not hasattr(calendar_module, name)
    assert callable(forcing_index_module.daily_forcing_index)
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
    assert not hasattr(pytree_utils_module, "concat_objects")
    assert "gravity" in physical_constants_module.PHYSICAL_CONSTANT_SETTINGS
    assert not hasattr(exchange_module, "VALID_EXCHANGE_FIELD_NAMES")
    assert "sea_surface_temperature" in field_names_module.VALID_EXCHANGE_FIELD_NAMES

    assert exchange_module.ExchangeField == str | tuple[str, str]
    assert hasattr(exchange_module, "RegridderFactory")

    clock_source = Path("vercor/clock.py").read_text(encoding="utf-8")
    calendar_source = Path("vercor/calendar.py").read_text(encoding="utf-8")
    time_selection_source = Path("vercor/time_selection.py").read_text(encoding="utf-8")
    forcing_index_path = Path("vercor/forcing_index.py")
    forcing_index_source = forcing_index_path.read_text(encoding="utf-8")
    runtime_time_source = Path("vercor/runtime/time.py").read_text(encoding="utf-8")
    runtime_validation_source = Path("vercor/runtime/validation.py").read_text(
        encoding="utf-8"
    )
    exchange_source = Path("vercor/exchange.py").read_text(encoding="utf-8")
    coupler_helpers_source = Path("vercor/setups/coupler_helpers.py").read_text(
        encoding="utf-8"
    )
    exchange_recipes_source = Path("vercor/setups/exchange_recipes.py").read_text(
        encoding="utf-8"
    )
    runtime_resources_source = Path("vercor/runtime/resources.py").read_text(
        encoding="utf-8"
    )
    runtime_cache_path = Path("vercor/runtime/cache.py")
    runtime_compilation_path = Path("vercor/runtime/compilation.py")
    regridder_base_source = Path("vercor/regridders/base.py").read_text(
        encoding="utf-8"
    )
    settings_source = Path("vercor/settings.py").read_text(encoding="utf-8")
    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")
    regridder_init = Path("vercor/regridders/__init__.py").read_text(encoding="utf-8")
    grid_masks_source = Path("vercor/grid_masks.py").read_text(encoding="utf-8")
    topology_source = Path("vercor/runtime/topology.py").read_text(encoding="utf-8")
    component_topology_source = Path("vercor/runtime/component_topology.py").read_text(
        encoding="utf-8"
    )
    jax_gcm_tools_source = Path("vercor/setups/external/jax_gcm_tools.py").read_text(
        encoding="utf-8"
    )

    assert forcing_index_path.exists()
    assert "mapped_day_in_month =" not in calendar_source
    assert 'if year_type == "360"' not in calendar_source
    assert "from vercor.forcing_index import daily_forcing_day_of_year" in (
        time_selection_source
    )
    assert "Protocol" not in time_selection_source
    assert "_custom_360_day_to_gregorian_day_of_year" not in time_selection_source
    assert "from vercor.forcing_index import daily_forcing_index" in (
        runtime_time_source
    )
    assert "from vercor.calendar import daily_forcing_index" not in runtime_time_source
    assert "from vercor.calendar import" in forcing_index_source
    assert "daily_forcing_index" in forcing_index_source
    assert not any(
        "vercor.forcing_index" in cycle
        for cycle in package_import_cycles("vercor", "vercor")
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
    assert "ExchangeField: TypeAlias" in exchange_source
    assert "RegridderFactory: TypeAlias" in exchange_source
    assert "ExchangeField = str | tuple[str, str]" not in coupler_helpers_source
    assert "RegridderFactory = Callable[" not in coupler_helpers_source
    assert "ExchangeField: TypeAlias" not in exchange_recipes_source
    assert "from vercor.exchange import ExchangeField" in coupler_helpers_source
    assert "from vercor.exchange import ExchangeField" in exchange_recipes_source
    assert not runtime_compilation_path.exists()
    assert not runtime_cache_path.exists()
    assert "from vercor.runtime.compilation import" not in runtime_resources_source
    assert "CompiledRuntime = Callable[" not in runtime_resources_source
    assert "CompiledRuntime" not in runtime_resources_source
    assert "RuntimeCompilationKey" not in runtime_resources_source
    assert "def _compute_has_identical_grids(" not in regridder_base_source
    assert "grids_identical(" in regridder_base_source
    assert "BilinearRectilinearInterpolator" not in regridder_base_source
    assert "ConservativeRectilinearRemapper" not in regridder_base_source
    assert "Protocol" not in regridder_base_source
    assert "SupportsScalarVectorInterpolation" not in regridder_base_source
    assert '"gravity": Settings(' not in settings_source
    assert "PHYSICAL_CONSTANT_SETTINGS" in settings_source
    assert "Incorrect component name" not in coupler_source
    assert "def validate_component_topology_names(" not in topology_source
    assert "def validate_component_topology_names(" in component_topology_source
    assert "make_rectilinear_grid" not in regridder_init
    assert "centers_to_edges" not in regridder_init
    assert "compute_land_mask" not in regridder_init
    assert "def compute_land_mask(" in grid_masks_source
    assert "def get_component(" not in grid_masks_source
    assert "def get_component(" not in topology_source
    assert "def get_component(" in component_topology_source
    assert "def compute_pressure_levels(" not in jax_gcm_tools_source
    assert "def get_altitudes_sigma_levels(" not in jax_gcm_tools_source
    assert "def mean_leaf(" not in jax_gcm_tools_source
    assert "def stack_objects(" not in jax_gcm_tools_source
    assert "def unwrap_leading_dims(" not in jax_gcm_tools_source


@pytest.mark.fast_always
def test_concrete_regridders_own_call_dispatch() -> None:
    regridder_base_source = Path("vercor/regridders/base.py").read_text(
        encoding="utf-8"
    )
    bilinear_source = Path("vercor/regridders/bilinear.py").read_text(encoding="utf-8")
    conservative_source = Path("vercor/regridders/conservative.py").read_text(
        encoding="utf-8"
    )

    assert "def __call__(" not in regridder_base_source
    assert "def _ensure_ready(" not in regridder_base_source
    assert "def __call__(" in bilinear_source
    assert "apply_vector" in bilinear_source
    assert "def __call__(" in conservative_source
    assert "def _ensure_ready(" not in conservative_source
    assert "apply_vector" not in conservative_source


@pytest.mark.fast_always
def test_setup_helper_and_external_output_ownership_boundaries() -> None:
    import vercor.diagnostics as diagnostics_module
    import vercor.host_arrays as host_arrays_module
    import vercor.setups.external.camulator as camulator_module
    import vercor.setups.external.camulator_contracts as camulator_contracts_module
    import vercor.setups.external.camulator_fields as camulator_fields_module
    import vercor.setups.external.camulator_land as camulator_land_module
    import vercor.setups.external.camulator_output as camulator_output_module
    import vercor.setups.external.camulator_runtime_settings as camulator_runtime_settings_module
    import vercor.setups.external.jax_gcm as jax_gcm_module
    import vercor.setups.external.jax_gcm_output as jax_gcm_output_module
    import vercor.output.adapters as output_adapters_module
    import vercor.output.period_averages as period_averages_module
    import vercor.output.period_files as period_files_module
    import vercor.setups.external.veros_output as veros_output_module
    import vercor.setups.external.veros_fluxes as veros_fluxes_module
    import vercor.setups.external.veros_gcm as veros_gcm_module
    import vercor.setups.external.veros_setup as veros_setup_module
    import vercor.setups.external.veros_state as veros_state_module

    assert callable(host_arrays_module.transposed_host_array)
    assert callable(diagnostics_module.component_vector_speed)
    assert callable(camulator_land_module.make_camulator_land)
    assert camulator_contracts_module.CAMULATOR_RUNTIME_FIELD_NAMES
    assert callable(camulator_fields_module.prepare_camulator_surface_forcing)
    assert callable(camulator_runtime_settings_module.configure_camulator_runtime)
    assert callable(output_adapters_module.ComponentOutputAdapter)
    assert callable(period_averages_module.PeriodAverageAccumulator)
    assert callable(period_files_module.write_period_average_netcdf)
    assert callable(jax_gcm_output_module.make_jax_gcm_output_adapter)
    assert callable(jax_gcm_output_module.record_jax_gcm_period_output)
    assert callable(veros_output_module.make_veros_output_adapter)
    assert callable(veros_output_module.record_veros_period_output)
    assert callable(camulator_output_module.make_camulator_output_adapter)
    assert callable(camulator_output_module.record_camulator_period_output)
    assert callable(veros_fluxes_module.compute_fluxes)
    assert callable(veros_output_module.extract_veros_output_snapshot)
    assert not hasattr(veros_output_module, "VerosOutputVariable")
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
    jax_gcm_state_source = Path("vercor/setups/external/jax_gcm_state.py").read_text(
        encoding="utf-8"
    )
    jax_gcm_runtime_source = Path(
        "vercor/setups/external/jax_gcm_runtime.py"
    ).read_text(encoding="utf-8")
    jax_gcm_fields_source = Path("vercor/setups/external/jax_gcm_fields.py").read_text(
        encoding="utf-8"
    )
    camulator_source = Path("vercor/setups/external/camulator.py").read_text(
        encoding="utf-8"
    )
    camulator_runtime_source = Path(
        "vercor/setups/external/camulator_runtime.py"
    ).read_text(encoding="utf-8")
    camulator_gcm_state_source = Path(
        "vercor/setups/external/camulator_gcm_state.py"
    ).read_text(encoding="utf-8")
    camulator_fields_source = Path(
        "vercor/setups/external/camulator_fields.py"
    ).read_text(encoding="utf-8")
    camulator_output_source = Path(
        "vercor/setups/external/camulator_output.py"
    ).read_text(encoding="utf-8")
    camulator_tensors_source = Path(
        "vercor/setups/external/camulator_tensors.py"
    ).read_text(encoding="utf-8")
    camulator_wind_filter_source = Path(
        "vercor/setups/external/camulator_wind_filter.py"
    ).read_text(encoding="utf-8")
    camulator_private_wind_filtering_path = Path(
        "vercor/setups/external/_camulator_wind_filtering.py"
    )
    camulator_private_wind_filtering_source = (
        camulator_private_wind_filtering_path.read_text(encoding="utf-8")
        if camulator_private_wind_filtering_path.exists()
        else ""
    )
    camulator_init_source = Path("vercor/setups/external/camulator_init.py").read_text(
        encoding="utf-8"
    )
    camulator_runtime_settings_source = Path(
        "vercor/setups/external/camulator_runtime_settings.py"
    ).read_text(encoding="utf-8")
    veros_gcm_source = Path("vercor/setups/external/veros_gcm.py").read_text(
        encoding="utf-8"
    )
    veros_gcm_state_source = Path(
        "vercor/setups/external/veros_gcm_state.py"
    ).read_text(encoding="utf-8")
    veros_runtime_source = Path("vercor/setups/external/veros_runtime.py").read_text(
        encoding="utf-8"
    )
    output_init_source = Path("vercor/output/__init__.py").read_text(encoding="utf-8")
    runtime_output_source = Path("vercor/output/runtime.py").read_text(encoding="utf-8")
    period_averages_source = Path("vercor/output/period_averages.py").read_text(
        encoding="utf-8"
    )
    output_adapters_source = Path("vercor/output/adapters.py").read_text(
        encoding="utf-8"
    )
    output_datasets_source = Path("vercor/output/datasets.py").read_text(
        encoding="utf-8"
    )
    period_files_source = Path("vercor/output/period_files.py").read_text(
        encoding="utf-8"
    )
    jax_gcm_output_source = Path("vercor/setups/external/jax_gcm_output.py").read_text(
        encoding="utf-8"
    )
    veros_output_source = Path("vercor/setups/external/veros_output.py").read_text(
        encoding="utf-8"
    )
    netcdf_output_source = Path("vercor/output/netcdf.py").read_text(encoding="utf-8")
    host_arrays_source = Path("vercor/host_arrays.py").read_text(encoding="utf-8")
    camulator_imports_source = Path(
        "vercor/setups/external/camulator_imports.py"
    ).read_text(encoding="utf-8")

    assert Path("vercor/setups/external/camulator_land.py").exists()
    assert Path("vercor/setups/external/camulator_runtime.py").exists()
    assert Path("vercor/setups/external/camulator_gcm_state.py").exists()
    assert not Path("vercor/output/jax_gcm.py").exists()
    assert Path("vercor/output/netcdf.py").exists()
    assert Path("vercor/output/period_averages.py").exists()
    assert Path("vercor/output/adapters.py").exists()
    assert Path("vercor/output/period_files.py").exists()
    assert Path("vercor/output/datasets.py").exists()
    assert Path("vercor/output/time.py").exists()
    assert Path("vercor/output/variables.py").exists()
    assert not Path("vercor/output/veros.py").exists()
    assert Path("vercor/setups/external/jax_gcm_fields.py").exists()
    assert Path("vercor/setups/external/jax_gcm_runtime.py").exists()
    assert Path("vercor/setups/external/camulator_output.py").exists()
    assert Path("vercor/setups/external/camulator_contracts.py").exists()
    assert Path("vercor/setups/external/camulator_fields.py").exists()
    assert Path("vercor/setups/external/camulator_runtime_settings.py").exists()
    assert Path("vercor/setups/external/camulator_wind_filter.py").exists()
    assert Path("vercor/setups/external/veros_fluxes.py").exists()
    assert Path("vercor/setups/external/veros_setup.py").exists()
    assert Path("vercor/setups/external/veros_state.py").exists()
    assert Path("vercor/setups/external/veros_runtime.py").exists()
    assert Path("vercor/setups/external/jax_gcm_state.py").exists()
    assert Path("vercor/setups/external/veros_gcm_state.py").exists()
    assert Path("vercor/setups/external/jax_gcm_output.py").exists()
    assert not Path("vercor/setups/external/period_averages.py").exists()
    assert Path("vercor/setups/external/veros_output.py").exists()
    assert not Path("vercor/setups/jax_array_helpers.py").exists()
    assert not Path("vercor/setups/data/camulator_land.py").exists()
    assert not Path("vercor/setups/external/windpp.py").exists()
    assert "from vercor.runtime.validation import" not in jax_gcm_source
    assert "class JAXGCMRuntimePayload" not in jax_gcm_source
    assert "class JAXGCMRuntimePayload" in jax_gcm_runtime_source
    assert "class JAXGCMSetupState" not in jax_gcm_source
    assert "class JAXGCMSetupState" in jax_gcm_state_source
    assert "def create_jax_gcm_runtime_payload(" in jax_gcm_runtime_source
    assert "def prefill_jax_gcm_runtime_fields(" in jax_gcm_runtime_source
    assert "def validate_jax_gcm_runtime_state(" in jax_gcm_runtime_source
    assert "def step_jax_gcm_runtime(" in jax_gcm_runtime_source
    assert "def record_jax_gcm_host_step(" in jax_gcm_runtime_source
    assert "def create_runtime_payload(" not in jax_gcm_source
    assert "def prefill_runtime_state_fields(" not in jax_gcm_source
    assert "def validate_runtime_state(" not in jax_gcm_source
    assert "def step_jax_gcm_runtime_callback(" not in jax_gcm_source
    assert "def create_jax_gcm_runtime_payload_callback(" not in jax_gcm_source
    assert "def prefill_jax_gcm_runtime_fields_callback(" not in jax_gcm_source
    assert "def validate_jax_gcm_runtime_state_callback(" not in jax_gcm_source
    assert "def step_jax_gcm_runtime_callback(" not in jax_gcm_state_source
    assert "def create_jax_gcm_runtime_payload_callback(" not in jax_gcm_state_source
    assert "def prefill_jax_gcm_runtime_fields_callback(" not in jax_gcm_state_source
    assert "def validate_jax_gcm_runtime_state_callback(" not in jax_gcm_state_source
    assert "def _step_jax_gcm_component_state(" not in jax_gcm_source
    assert "def _record_jax_gcm_host_step(" not in jax_gcm_source
    assert "def asfloat(" not in jax_gcm_source
    assert "def cleanup_surface_temperature_fields(" not in jax_gcm_source
    assert "def prepare_surface_temperature_forcing(" not in jax_gcm_source
    assert "def map_jcm_output_fields(" not in jax_gcm_source
    assert "def cleanup_surface_temperature_fields(" in jax_gcm_fields_source
    assert "def _should_write_output(" not in jax_gcm_source
    assert "def _write_output(" not in jax_gcm_source
    assert "os.environ[" not in camulator_source
    assert "import torch" not in camulator_source
    assert "import xarray" not in camulator_source
    assert "RectilinearGrid" not in camulator_source
    assert "CamulatorRuntimeCursor" not in camulator_source
    assert "assign_model_timestep_alignment" not in camulator_source
    assert "seed_grid_field_defaults" not in camulator_source
    assert "def configure_camulator_runtime(" in camulator_runtime_settings_source
    assert "class CAMulatorGCMSetupState" not in camulator_source
    assert "class CAMulatorGCMSetupState" in camulator_gcm_state_source
    assert "def coerce_camulator_datetime(" in camulator_runtime_source
    assert "def run_camulator_prediction_block(" in camulator_runtime_source
    assert "def step_camulator_runtime(" in camulator_runtime_source
    assert "def run_camulator_prediction_block(" not in camulator_source
    assert "def prepare_camulator_surface_forcing(" not in camulator_source
    assert "def map_camulator_prediction_arrays(" not in camulator_source
    assert "def prepare_camulator_surface_forcing(" in camulator_fields_source
    assert "def map_camulator_prediction_arrays(" in camulator_fields_source
    assert "def torch_tensor_from_jax_array(" not in camulator_source
    assert "def torch_tensor_from_jax_array(" in camulator_tensors_source
    assert "def add_init_noise(" not in camulator_source
    assert "def add_init_noise(" in camulator_init_source
    assert "def _credit_output_functions(" not in camulator_source
    assert "def _write_camulator_prediction_output(" not in camulator_source
    assert "class CustomGlobalFourDegree" not in veros_gcm_source
    assert "class VerosGCMSetupState" not in veros_gcm_source
    assert "class VerosGCMSetupState" in veros_gcm_state_source
    assert "def compute_fluxes(" not in veros_gcm_source
    assert "def copy_state(" not in veros_gcm_source
    assert "def set_variable(" not in veros_gcm_source
    assert "def step_veros_runtime(" in veros_runtime_source
    assert "compute_fluxes(" in veros_runtime_source
    assert "apply_veros_forcing_fields(" in veros_runtime_source
    assert "advance_veros_substeps(" in veros_runtime_source
    assert "compute_fluxes(" not in veros_gcm_source
    assert "apply_veros_forcing_fields(" not in veros_gcm_source
    assert "advance_veros_substeps(" not in veros_gcm_source
    assert "import h5netcdf" not in veros_gcm_source
    assert "import h5netcdf" not in veros_runtime_source
    assert "import h5netcdf" not in jax_gcm_output_source
    assert "import h5netcdf" not in veros_output_source
    assert "import h5netcdf" not in period_files_source
    assert "import numpy" not in veros_gcm_source
    assert "import numpy" not in veros_runtime_source
    assert "import h5netcdf" in netcdf_output_source
    assert "import xarray" not in runtime_output_source
    assert ".to_netcdf(" not in runtime_output_source
    assert "from vercor.output.netcdf import write_netcdf_dataset" in (
        runtime_output_source
    )
    assert "from vercor.output.netcdf import write_netcdf_dataset" in (
        period_files_source
    )
    assert "from vercor.output.netcdf import write_netcdf_dataset" not in (
        jax_gcm_output_source
    )
    assert "from vercor.output.netcdf import write_netcdf_dataset" not in (
        veros_output_source
    )
    assert "write_netcdf_dataset(" in runtime_output_source
    assert "write_netcdf_dataset(" in period_files_source
    snapshot_output_source = runtime_output_source.split(
        "def write_coupler_component_snapshots(",
        1,
    )[1]
    assert "component_snapshot_writer(" in snapshot_output_source
    assert "field_spec.outputs" not in snapshot_output_source
    assert ".data.get(" not in snapshot_output_source
    assert "RuntimeComponentView.from_component_state" not in snapshot_output_source
    assert "import numpy" not in period_averages_source
    assert "import numpy" not in period_files_source
    assert "import numpy" not in jax_gcm_output_source
    assert "import numpy" not in veros_output_source
    assert "import jax.numpy as jnp" in period_averages_source
    assert "import jax.numpy as jnp" in veros_output_source
    assert "def time_coordinate_variable(" in output_datasets_source
    assert "def used_dimension_names(" in output_datasets_source
    assert "def period_mean_output_variables(" in period_averages_source
    assert "def accumulate_output_variables(" not in period_averages_source
    assert "def write_period_average_netcdf(" in period_files_source
    assert "class ComponentOutputAdapter" in output_adapters_source
    assert "accumulate_output_variables(" not in output_adapters_source
    assert "self._accumulator.add_samples(" in output_adapters_source
    assert "def record_snapshot(" in output_adapters_source
    assert "def register_component_snapshot_writer(" in output_adapters_source
    assert "def record_period_average_if_due(" in output_adapters_source
    assert "period_mean_output_variables(" in output_adapters_source
    assert "write_period_average_netcdf(" in output_adapters_source
    assert "should_write_period_output(" in output_adapters_source
    assert "MeanVariablesBuilder" not in period_files_source
    assert "CoordinateVariablesBuilder" not in period_files_source
    assert "DataVariablesBuilder" not in period_files_source
    assert "accumulate_output_variables(" not in jax_gcm_output_source
    assert "period_mean_output_variables(" not in jax_gcm_output_source
    assert "write_period_average_netcdf(" not in jax_gcm_output_source
    assert "def make_jax_gcm_output_adapter(" in jax_gcm_output_source
    assert "def record_jax_gcm_period_output(" in jax_gcm_output_source
    assert "def write_jax_gcm_snapshot_output(" in jax_gcm_output_source
    assert "time_coordinate_variable(" in jax_gcm_output_source
    assert "accumulate_output_variables(" not in veros_output_source
    assert "period_mean_output_variables(" not in veros_output_source
    assert "write_period_average_netcdf(" not in veros_output_source
    assert "def make_veros_output_adapter(" in veros_output_source
    assert "def record_veros_period_output(" in veros_output_source
    assert "def write_veros_snapshot_output(" in veros_output_source
    assert "time_coordinate_variable(" in veros_output_source
    assert "used_dimension_names(" in veros_output_source
    assert "def make_camulator_output_adapter(" in camulator_output_source
    assert "def record_camulator_period_output(" in camulator_output_source
    assert "def write_camulator_snapshot_output(" in camulator_output_source
    assert "write_period_average_if_due(" not in jax_gcm_runtime_source
    assert "write_period_average_if_due(" not in veros_runtime_source
    assert "write_period_average_if_due(" not in camulator_runtime_source
    assert "def __getattr__(" not in output_init_source
    assert "def __dir__(" not in output_init_source
    assert "_RUNTIME_EXPORTS" not in output_init_source
    assert "from vercor.output.runtime import (" in output_init_source
    assert "write_coupler_runtime_outputs" in output_init_source
    assert "write_runtime_component_view_to_netcdf" in output_init_source
    assert "def array_to_host(" in host_arrays_source
    assert "def host_int64_array(" in host_arrays_source
    assert "from vercor.setups.external.camulator_wind_filter import" in (
        camulator_imports_source
    )
    assert camulator_private_wind_filtering_path.exists()
    assert (
        "import vercor.setups.external._camulator_wind_filtering as _wind_filtering"
        in camulator_wind_filter_source
    )
    assert "torch.nn.functional" not in camulator_wind_filter_source
    assert "def build_wind_filter_artifacts(" in camulator_private_wind_filtering_source
    assert "def apply_wind_filter_to_tensor(" in camulator_private_wind_filtering_source
    assert "F.conv2d(" in camulator_private_wind_filtering_source
    assert "_jax_gcm_fields._map_jcm_output_fields(" not in jax_gcm_runtime_source
    assert "_camulator_fields._prepare_camulator_surface_forcing(" not in (
        camulator_runtime_source
    )
    assert "_camulator_tensors._torch_tensor_from_jax_array(" not in (
        camulator_runtime_source
    )
    assert "_veros_state._prepare_surface_forcing_fields(" not in veros_runtime_source
    assert "_veros_state._advance_veros_substeps(" not in veros_runtime_source
    import vercor.setups.external as external_module

    assert "JAXGCMRuntimePayload" not in external_module._LAZY_EXPORTS


@pytest.mark.fast_always
def test_external_adapter_all_exports_are_public() -> None:
    for path in Path("vercor/setups/external").glob("*.py"):
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
        "vercor/setups/external/jax_gcm_runtime.py": (
            "JAXGCMSetupState",
            "vercor.setups.external.jax_gcm_state",
        ),
        "vercor/setups/external/veros_runtime.py": (
            "VerosGCMSetupState",
            "vercor.setups.external.veros_gcm_state",
        ),
        "vercor/setups/external/camulator_runtime.py": (
            "CAMulatorGCMSetupState",
            "vercor.setups.external.camulator_gcm_state",
        ),
    }

    for path_name, (state_name, state_module) in runtime_sources.items():
        source = Path(path_name).read_text(encoding="utf-8")
        assert "Protocol" not in source
        assert f"from {state_module} import {state_name}" in source
        assert f"state: {state_name}" in source or f'state: "{state_name}"' in source


@pytest.mark.fast_always
def test_jax_gcm_factory_binds_runtime_hooks_directly() -> None:
    source = Path("vercor/setups/external/jax_gcm.py").read_text(encoding="utf-8")
    state_source = Path("vercor/setups/external/jax_gcm_state.py").read_text(
        encoding="utf-8"
    )
    factory_source = source.split("def make_jax_gcm(", 1)[1]

    assert "from functools import partial" in source
    assert "def step_jax_gcm_runtime_callback(" not in state_source
    assert "def create_jax_gcm_runtime_payload_callback(" not in state_source
    assert "def prefill_jax_gcm_runtime_fields_callback(" not in state_source
    assert "def validate_jax_gcm_runtime_state_callback(" not in state_source
    assert "step=partial(_jax_gcm_runtime.step_jax_gcm_component, state)" in (
        factory_source
    )
    assert "_jax_gcm_runtime.create_jax_gcm_runtime_payload" in factory_source
    assert "_jax_gcm_runtime.prefill_jax_gcm_runtime_fields" in factory_source
    assert "_jax_gcm_runtime.validate_jax_gcm_runtime_state" in factory_source
    assert "_jax_gcm_state.step_jax_gcm_runtime_callback" not in factory_source
    assert "_jax_gcm_state.create_jax_gcm_runtime_payload_callback" not in (
        factory_source
    )
    assert "_jax_gcm_state.prefill_jax_gcm_runtime_fields_callback" not in (
        factory_source
    )
    assert "_jax_gcm_state.validate_jax_gcm_runtime_state_callback" not in (
        factory_source
    )
    assert "lambda fields" not in factory_source
    assert "lambda component" not in factory_source


@pytest.mark.fast_always
def test_jax_gcm_average_writer_bypasses_xarray_adapter() -> None:
    source = Path("vercor/setups/external/jax_gcm_output.py").read_text(
        encoding="utf-8"
    )

    assert "import xarray" not in source
    assert ".to_xarray(" not in source


@pytest.mark.fast_always
def test_external_package_has_no_top_level_import_cycles() -> None:
    assert (
        package_import_cycles("vercor/setups/external", "vercor.setups.external") == []
    )


@pytest.mark.fast_always
def test_output_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/output", "vercor.output") == []


@pytest.mark.fast_always
def test_veros_runtime_settings_imports_runtime_settings_lazily() -> None:
    source = Path("vercor/setups/external/veros_runtime_settings.py").read_text(
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
        if path.name.startswith("run_"):
            assert "ExchangeSpec(" in source, path


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
    diagnostics_fields_source = Path("vercor/diagnostics/fields.py").read_text(
        encoding="utf-8"
    )
    assert ".data.get(" not in diagnostics_fields_source
    assert "getattr(" not in diagnostics_fields_source
    assert "def view_field_candidates(" not in diagnostics_fields_source
    assert "def view_field(" not in diagnostics_fields_source
    assert "runtime_field(" in diagnostics_fields_source


@pytest.mark.fast_always
def test_veros_setup_state_does_not_keep_one_line_step_wrapper() -> None:
    gcm_source = Path("vercor/setups/external/veros_gcm.py").read_text(encoding="utf-8")
    source = Path("vercor/setups/external/veros_gcm_state.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    setup_state_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VerosGCMSetupState"
    )

    assert "def advance_veros_model_step(" not in source
    assert '"advance_veros_model_step"' not in source
    assert "partial(" in source
    assert "_veros_state.pure," in source
    assert "step" not in {
        node.name
        for node in setup_state_class.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "from functools import partial" in gcm_source
    assert "import vercor.setups.external.veros_runtime as _veros_runtime" in gcm_source
    assert "step=partial(_veros_runtime.step_veros_runtime, state)," in gcm_source


@pytest.mark.fast_always
def test_camulator_adapters_share_runtime_cursor_state_transition_helper() -> None:
    for path in (Path("vercor/setups/external/camulator_gcm_state.py"),):
        source = path.read_text(encoding="utf-8")
        assert "CamulatorRuntimeCursor" in source, path
        assert "runtime_forcing_index(" not in source, path
        assert "timestep_counter += 1" not in source, path

    for path in (
        Path("vercor/setups/external/camulator.py"),
        Path("vercor/setups/external/camulator_land.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "runtime_forcing_index(" not in source, path
        assert "timestep_counter += 1" not in source, path


@pytest.mark.fast_always
def test_camulator_gcm_factory_passes_runtime_step_directly() -> None:
    gcm_source = Path("vercor/setups/external/camulator.py").read_text(encoding="utf-8")
    state_source = Path("vercor/setups/external/camulator_gcm_state.py").read_text(
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
        "import vercor.setups.external.camulator_runtime as _camulator_runtime"
        in gcm_source
    )
    assert (
        "step=partial(_camulator_runtime.step_camulator_runtime, state)," in gcm_source
    )


@pytest.mark.fast_always
def test_camulator_state_facade_is_removed() -> None:
    focused_modules = (
        Path("vercor/setups/external/camulator_imports.py"),
        Path("vercor/setups/external/camulator_forcing.py"),
        Path("vercor/setups/external/camulator_tensors.py"),
        Path("vercor/setups/external/camulator_stepper.py"),
        Path("vercor/setups/external/camulator_init.py"),
        Path("vercor/setups/external/camulator_gcm_state.py"),
    )
    for path in focused_modules:
        assert path.exists(), path

    assert not Path("vercor/setups/external/camulator_state.py").exists()
    gcm_state_source = Path("vercor/setups/external/camulator_gcm_state.py").read_text(
        encoding="utf-8"
    )
    stepper_source = Path("vercor/setups/external/camulator_stepper.py").read_text(
        encoding="utf-8"
    )
    assert "accessor_state" not in gcm_state_source
    assert "StateVariableAccessor" not in stepper_source
    assert "def step(" not in stepper_source
    assert "def get_state_var(" not in stepper_source
    assert "def set_state_var(" not in stepper_source


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
    assert "_lon_src_2d" not in source
    assert "_lat_src_2d" not in source


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
        "import vercor.setups.external",
        "import vercor.setups.data",
        "import vercor.setups.jcm_setup_helpers",
        "from vercor.setups.external import __all__",
        "from vercor.setups.data import __all__",
        "import vercor.setups.data.era5_land",
        "from vercor.setups.data import make_era5_land",
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
