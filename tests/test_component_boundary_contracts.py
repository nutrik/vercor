from __future__ import annotations

import ast

import pytest

from tests._architecture_support import (
    class_body_source,
    package_import_cycles,
    source_for,
)


def _imported_names_from(path: str, module: str) -> set[str]:
    """Return names imported from one module in a Python source file."""

    tree = ast.parse(source_for(path))
    imported_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            imported_names.update(alias.name for alias in node.names)
    return imported_names


def test_runtime_component_protocol_excludes_setup_data_storage() -> None:
    protocol_source = class_body_source(
        "vercor/components/_protocols.py",
        "ComponentRuntimeProtocol",
    )

    assert "data:" not in protocol_source
    assert "setup_metadata" not in protocol_source


def test_component_runtime_helpers_keep_private_protocol_boundary() -> None:
    helper_paths = (
        "vercor/components/_runtime_fields.py",
        "vercor/components/_runtime_validation.py",
        "vercor/components/_runtime_access.py",
        "vercor/components/_lifecycle_api.py",
        "vercor/components/_callable_wrappers.py",
        "vercor/components/runtime_execution.py",
    )

    for path in helper_paths:
        source = source_for(path)
        assert "from vercor.components.base import Component" not in source, path
        assert "vercor.components._protocols" in source, path

    components_source = source_for("vercor/components/__init__.py")
    assert "_protocols" not in components_source


@pytest.mark.fast_always
def test_component_execution_protocols_are_private_structural_contracts() -> None:
    protocol_source = source_for("vercor/components/_protocols.py")
    execution_protocol_source = class_body_source(
        "vercor/components/_protocols.py",
        "ComponentExecutionProtocol",
    )
    host_execution_protocol_source = class_body_source(
        "vercor/components/_protocols.py",
        "HostRuntimeExecutionProtocol",
    )

    assert "class ComponentExecutionProtocol" in protocol_source
    assert "@runtime_checkable\nclass HostRuntimeExecutionProtocol" in protocol_source
    assert "def step_runtime_state(" in execution_protocol_source
    assert "def step_host_runtime_state(" in host_execution_protocol_source
    assert "from vercor.components.base import Component" not in protocol_source
    assert "from vercor.components.host import HostRuntimeComponent" not in (
        protocol_source
    )


@pytest.mark.fast_always
def test_public_lifecycle_hook_types_are_owned_by_component_contracts() -> None:
    hook_names = {
        "ComponentInitializeHook",
        "ComponentCreatePayloadHook",
        "ComponentPrefillHook",
        "ComponentValidateHook",
    }
    contracts_source = source_for("vercor/components/contracts.py")
    lifecycle_source = source_for("vercor/components/_lifecycle.py")

    for hook_name in hook_names:
        assert f"{hook_name} =" in contracts_source
        assert f"{hook_name} =" not in lifecycle_source

    for path in (
        "vercor/components/base.py",
        "vercor/components/host.py",
        "vercor/components/factories.py",
    ):
        private_imports = _imported_names_from(path, "vercor.components._lifecycle")
        public_imports = _imported_names_from(path, "vercor.components.contracts")
        assert hook_names.isdisjoint(private_imports), path
        assert hook_names.issubset(public_imports), path


@pytest.mark.fast_always
def test_lifecycle_storage_uses_private_typed_owner_boundary() -> None:
    lifecycle_source = source_for("vercor/components/_lifecycle.py")
    protocol_source = source_for("vercor/components/_protocols.py")

    assert "class ComponentLifecycleOwner(Protocol)" in lifecycle_source
    assert "component: ComponentLifecycleOwner" in lifecycle_source
    assert "component: Any" not in lifecycle_source
    assert "_lifecycle_hooks: ComponentLifecycleHooks" in protocol_source
    assert "_lifecycle_hooks: Any" not in protocol_source


@pytest.mark.fast_always
def test_callable_wrapper_module_owns_callable_component_definition() -> None:
    callable_source = source_for("vercor/components/_callable_wrappers.py")
    base_source = source_for("vercor/components/base.py")
    host_source = source_for("vercor/components/host.py")

    assert "class _CallableComponentDefinition" in callable_source
    assert "def _callable_component_definition(" in callable_source
    assert "lifecycle_hooks: ComponentLifecycleHooks" in callable_source
    assert "initialize: ComponentInitializeHook | None" not in class_body_source(
        "vercor/components/_callable_wrappers.py",
        "_CallableComponentDefinition",
    )
    assert "def create_runtime_payload(" not in callable_source
    assert "component._lifecycle_hooks.create_runtime_payload" not in callable_source
    assert "field_spec=_ComponentFieldSpec(" not in base_source
    assert "field_spec=ComponentFieldSpec(" not in host_source


def test_components_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/components", "vercor.components") == []
