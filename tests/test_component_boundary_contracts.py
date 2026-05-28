from __future__ import annotations

from tests._architecture_support import (
    class_body_source,
    package_import_cycles,
    source_for,
)


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
    )

    for path in helper_paths:
        source = source_for(path)
        assert "from vercor.components.base import Component" not in source, path
        assert "vercor.components._protocols" in source, path

    components_source = source_for("vercor/components/__init__.py")
    assert "_protocols" not in components_source


def test_components_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/components", "vercor.components") == []
