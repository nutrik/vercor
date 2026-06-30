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


def test_component_runtime_helpers_do_not_keep_annotation_only_protocol_layer() -> None:
    helper_paths = (
        "vercor/components/_runtime_fields.py",
        "vercor/components/_runtime_validation.py",
        "vercor/components/_lifecycle_api.py",
        "vercor/components/_callable_wrappers.py",
        "vercor/components/runtime_execution.py",
    )

    for path in helper_paths:
        source = source_for(path)
        if path == "vercor/components/runtime_execution.py":
            assert "HostRuntimeExecutionProtocol" in source, path
        else:
            assert "vercor.components._protocols" not in source, path
        assert "if TYPE_CHECKING:" in source, path

    protocol_source = source_for("vercor/components/_protocols.py")
    assert protocol_source.count("class ") == 1
    components_source = source_for("vercor/components/__init__.py")
    assert "_protocols" not in components_source


@pytest.mark.fast_always
def test_host_runtime_execution_protocol_is_only_private_structural_contract() -> None:
    protocol_source = source_for("vercor/components/_protocols.py")
    host_execution_protocol_source = class_body_source(
        "vercor/components/_protocols.py",
        "HostRuntimeExecutionProtocol",
    )

    assert "class ComponentRuntimeProtocol" not in protocol_source
    assert "class ComponentAuthoringProtocol" not in protocol_source
    assert "class ComponentExecutionProtocol" not in protocol_source
    assert "@runtime_checkable\nclass HostRuntimeExecutionProtocol" in protocol_source
    assert "def step_runtime_state(" in host_execution_protocol_source
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
    assert "_lifecycle_hooks: ComponentLifecycleHooks" in lifecycle_source
    assert "_lifecycle_hooks: Any" not in lifecycle_source
    assert "_lifecycle_hooks" not in protocol_source


@pytest.mark.fast_always
def test_callable_wrapper_module_does_not_need_request_dataclass() -> None:
    callable_source = source_for("vercor/components/_callable_wrappers.py")
    base_source = source_for("vercor/components/base.py")
    host_source = source_for("vercor/components/host.py")

    assert "class _CallableComponentDefinition" not in callable_source
    assert "def _callable_component_definition(" not in callable_source
    assert "lifecycle_hooks: ComponentLifecycleHooks" in callable_source
    assert "def create_runtime_payload(" not in callable_source
    assert "component._lifecycle_hooks.create_runtime_payload" not in callable_source
    assert "field_spec=_ComponentFieldSpec(" not in base_source
    assert "field_spec=ComponentFieldSpec(" not in host_source


def test_components_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/components", "vercor.components") == []
