from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from vercor.components.contracts import (
    AuthorFieldValues as _AuthorFieldValues,
    ComponentFieldSpec as _ComponentFieldSpec,
    ComponentStepReturn as _ComponentStepReturn,
    FieldNames as _FieldNames,
)
import vercor.components._runtime_fields as _runtime_field_adapters
import vercor.components._runtime_validation as _runtime_field_validation
from vercor.dtypes import PrecisionPolicy
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.runtime.state import RuntimeComponentState


class ComponentRuntimeAccessMixin:
    """Component-facing runtime field read, update, prefill, and validation API."""

    def _runtime_access_component(self) -> "Component":
        """Return this mixin instance as the concrete component type."""

        return cast("Component", self)

    def runtime_fields(
        self,
        component_state: "RuntimeComponentState",
    ) -> dict[str, RuntimeArray]:
        """Return runtime data fields as a plain name-to-array mapping."""

        return _runtime_field_adapters.runtime_fields(
            self._runtime_access_component(), component_state
        )

    def runtime_field(
        self,
        component_state: "RuntimeComponentState",
        name: str,
    ) -> RuntimeArray:
        """Return one runtime data field with a component-oriented error."""

        return _runtime_field_adapters.runtime_field(
            self._runtime_access_component(), component_state, name
        )

    def has_runtime_field(
        self,
        component_state: "RuntimeComponentState",
        name: str,
    ) -> bool:
        """Return whether one runtime data field exists."""

        return _runtime_field_adapters.has_runtime_field(
            self._runtime_access_component(), component_state, name
        )

    def runtime_field_or(
        self,
        component_state: "RuntimeComponentState",
        name: str,
        default: object,
        policy: PrecisionPolicy = None,
    ) -> RuntimeArray:
        """Return one runtime field or a grid-shaped/default array fallback."""

        return _runtime_field_adapters.runtime_field_or(
            self._runtime_access_component(),
            component_state,
            name,
            default,
            policy,
        )

    def runtime_field_or_zeros_like(
        self,
        component_state: "RuntimeComponentState",
        name: str,
        like: str | RuntimeArray,
    ) -> RuntimeArray:
        """Return one runtime field or zeros matching another field/array."""

        return _runtime_field_adapters.runtime_field_or_zeros_like(
            self._runtime_access_component(),
            component_state,
            name,
            like,
        )

    def with_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
        fields: Mapping[str, RuntimeArray],
    ) -> "RuntimeComponentState":
        """Return ``component_state`` with existing runtime data fields updated."""

        return _runtime_field_adapters.with_runtime_fields(
            self._runtime_access_component(),
            component_state,
            fields,
        )

    def apply_step_result(
        self,
        component_state: "RuntimeComponentState",
        result: _ComponentStepReturn,
    ) -> "RuntimeComponentState":
        """Apply a field mapping or ``ComponentStepResult`` to runtime state."""

        return _runtime_field_adapters.apply_step_result(
            self._runtime_access_component(), component_state, result
        )

    def require_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
        *names: str,
    ) -> None:
        """Validate that named runtime data fields use canonical grid layout."""

        _runtime_field_validation.require_runtime_fields(
            self._runtime_access_component(), component_state, *names
        )

    def prefill_runtime_fields(
        self,
        data: dict[str, RuntimeArray],
        field_spec: _ComponentFieldSpec | None = None,
        *,
        outputs: _FieldNames = (),
        default_fields: _AuthorFieldValues = None,
        policy: PrecisionPolicy = None,
    ) -> None:
        """Prefill a mutable runtime data mapping with declared fields."""

        _runtime_field_adapters.prefill_runtime_fields(
            self._runtime_access_component(),
            data,
            field_spec,
            outputs=outputs,
            default_fields=default_fields,
            policy=policy,
        )


__all__ = ["ComponentRuntimeAccessMixin"]
