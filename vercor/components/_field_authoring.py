from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from vercor.components.contracts import (
    AuthorFieldValues as _AuthorFieldValues,
    ComponentFieldSpec as _ComponentFieldSpec,
    FieldNames as _FieldNames,
)
from vercor.components._contracts import (
    normalize_author_field_values as _normalize_author_field_values,
)
from vercor.components._field_names import unique_field_names as _unique_field_names
from vercor.dtypes import PrecisionPolicy, jax_full, jax_zeros
from vercor.exceptions import ComponentError
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray


class ComponentFieldAuthoringMixin:
    """Author-facing field declaration, setup seeding, and settings helpers."""

    name: str
    grid: RectilinearGrid
    data: dict[str, RuntimeArray]
    settings: VercorSettings
    _field_spec: _ComponentFieldSpec

    def declare_fields(
        self,
        field_spec: _ComponentFieldSpec | None = None,
        *,
        inputs: _FieldNames = (),
        outputs: _FieldNames = (),
        default_fields: _AuthorFieldValues = None,
    ) -> _ComponentFieldSpec:
        """Declare runtime data fields for subclasses using author-facing names."""

        declared = field_spec or _ComponentFieldSpec(
            inputs=inputs,
            outputs=outputs,
            default_fields=default_fields or {},
        )
        self._field_spec = _ComponentFieldSpec(
            inputs=declared.inputs,
            outputs=declared.outputs,
            default_fields=_normalize_author_field_values(
                component_name=self.name,
                grid=self.grid,
                fields=declared.default_fields,
                policy=self.settings,
            )
            or {},
        )
        return self._field_spec

    @property
    def field_spec(self) -> _ComponentFieldSpec:
        """Return this component's declared author-facing runtime field contract."""

        return self._field_spec

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return setup-time field names in insertion order."""

        return tuple(self.data)

    def update_settings(self, **values: object) -> Self:
        """Update component settings by name and return this component."""

        for setting_name, setting_value in values.items():
            self.settings.set_value(setting_name, setting_value)
        return self

    def grid_field_defaults(
        self,
        names: _FieldNames,
        value: object = 0.0,
        overrides: _AuthorFieldValues = None,
        policy: PrecisionPolicy = None,
    ) -> dict[str, RuntimeArray]:
        """Return grid-shaped default fields for named runtime data fields."""

        field_names = _unique_field_names(names)
        defaults: dict[str, object] = {field_name: value for field_name in field_names}
        for field_name, field_value in (overrides or {}).items():
            if field_name not in defaults:
                raise ComponentError(
                    f"Default override field '{field_name}' is not declared for "
                    f"component '{self.name}'."
                )
            defaults[field_name] = field_value

        return (
            _normalize_author_field_values(
                component_name=self.name,
                grid=self.grid,
                fields=defaults,
                policy=self.settings if policy is None else policy,
            )
            or {}
        )

    def seed_field(
        self: Self,
        name: str,
        value: object,
        policy: PrecisionPolicy = None,
    ) -> Self:
        """Seed one setup-time grid field and return this component."""

        return self.seed_fields({name: value}, policy=policy)

    def seed_fields(
        self: Self,
        fields: Mapping[str, object],
        policy: PrecisionPolicy = None,
    ) -> Self:
        """Seed setup-time grid fields and return this component."""

        field_updates = _normalize_author_field_values(
            component_name=self.name,
            grid=self.grid,
            fields=fields,
            policy=self.settings if policy is None else policy,
        )
        self.data.update(field_updates or {})
        return self

    def seed_declared_defaults(
        self: Self,
        policy: PrecisionPolicy = None,
    ) -> Self:
        """Seed this component's declared default fields and return itself."""

        default_fields = self._field_spec.default_fields
        if default_fields:
            self.seed_fields(default_fields, policy=policy)
        return self

    def seed_zero_field(
        self: Self,
        name: str,
        policy: PrecisionPolicy = None,
    ) -> Self:
        """Seed one grid-shaped zero field and return this component."""

        return self.seed_field(
            name,
            jax_zeros(
                self.grid.shape,
                self.settings if policy is None else policy,
            ),
        )

    def seed_zero_fields(
        self: Self,
        names: _FieldNames,
        policy: PrecisionPolicy = None,
    ) -> Self:
        """Seed multiple grid-shaped zero fields and return this component."""

        for name in names:
            self.seed_zero_field(name, policy)
        return self

    def seed_constant_field(
        self: Self,
        name: str,
        value: object,
        policy: PrecisionPolicy = None,
    ) -> Self:
        """Seed one grid-shaped constant field and return this component."""

        return self.seed_field(
            name,
            jax_full(
                self.grid.shape,
                value,
                self.settings if policy is None else policy,
            ),
        )


__all__ = ["ComponentFieldAuthoringMixin"]
