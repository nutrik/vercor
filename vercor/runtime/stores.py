from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import jax
import jax.numpy as jnp

from vercor.pytree import PyTreeNodeMixin
from vercor.types import RuntimeArray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeFieldStore(PyTreeNodeMixin):
    """Immutable named array store used by the runtime."""

    pytree_children = ("values",)
    pytree_aux_data = ("field_names",)

    field_names: tuple[str, ...]
    values: tuple[RuntimeArray, ...]
    field_indices: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build static lookup metadata for name-based runtime access."""

        object.__setattr__(
            self,
            "field_indices",
            {name: index for index, name in enumerate(self.field_names)},
        )

    def _pytree_post_unflatten(self) -> None:
        """Restore derived lookup metadata after PyTree unflattening."""

        self.__post_init__()

    @classmethod
    def empty(cls) -> "RuntimeFieldStore":
        """Create an empty field store."""

        return cls(field_names=(), values=())

    @classmethod
    def from_mapping(cls, fields: Mapping[str, RuntimeArray]) -> "RuntimeFieldStore":
        """Create a field store from a mapping while preserving insertion order."""

        return cls(
            field_names=tuple(fields.keys()),
            values=tuple(jnp.array(value, copy=True) for value in fields.values()),
        )

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is present in this store."""

        return isinstance(name, str) and name in self.field_indices

    def to_mapping(self) -> dict[str, RuntimeArray]:
        """Return this store as a plain name-to-array mapping."""

        return dict(zip(self.field_names, self.values, strict=True))

    def get(self, name: str) -> RuntimeArray:
        """Return a field by name."""

        try:
            index = self.field_indices[name]
        except KeyError as exc:
            raise KeyError(f"Runtime field {name!r} not found") from exc
        return self.values[index]

    def get_or(self, name: str, default: RuntimeArray) -> RuntimeArray:
        """Return a field by name, or ``default`` when it is absent."""

        if name in self:
            return self.get(name)
        return jnp.asarray(default)

    def get_or_zeros_like(
        self,
        name: str,
        like: str | RuntimeArray,
    ) -> RuntimeArray:
        """Return a field by name, or zeros matching another field or array."""

        if name in self:
            return self.get(name)
        reference = self.get(like) if isinstance(like, str) else like
        return jnp.zeros_like(jnp.asarray(reference))

    def set(self, name: str, value: RuntimeArray) -> "RuntimeFieldStore":
        """Return a new store with ``name`` replaced or appended."""

        if name not in self:
            value_array = jnp.array(value, copy=True)
            return RuntimeFieldStore(
                field_names=(*self.field_names, name),
                values=(*self.values, value_array),
            )

        return self.replace(name, value)

    def set_many(self, fields: Mapping[str, RuntimeArray]) -> "RuntimeFieldStore":
        """Return a new store with multiple fields replaced or appended."""

        if not fields:
            return self

        values = list(self.values)
        appended_names: list[str] = []
        appended_values: list[RuntimeArray] = []
        for field_name, field_value in fields.items():
            if field_name in self.field_indices:
                index = self.field_indices[field_name]
                current = values[index]
                values[index] = jnp.array(
                    field_value,
                    dtype=jnp.asarray(current).dtype,
                    copy=True,
                )
            else:
                appended_names.append(field_name)
                appended_values.append(jnp.array(field_value, copy=True))

        return RuntimeFieldStore(
            field_names=(*self.field_names, *appended_names),
            values=(*values, *appended_values),
        )

    def replace(self, name: str, value: RuntimeArray) -> "RuntimeFieldStore":
        """Return a new store with an existing field replaced."""

        if name not in self:
            raise KeyError(f"Runtime field {name!r} not found")

        values = tuple(
            (
                jnp.array(value, dtype=jnp.asarray(current).dtype, copy=True)
                if field_name == name
                else current
            )
            for field_name, current in zip(self.field_names, self.values)
        )
        return RuntimeFieldStore(field_names=self.field_names, values=values)

    def replace_many(
        self,
        fields: Mapping[str, RuntimeArray],
    ) -> "RuntimeFieldStore":
        """Return a new store with multiple existing fields replaced."""

        for field_name in fields:
            if field_name not in self:
                raise KeyError(f"Runtime field {field_name!r} not found")
        return self.set_many(fields)
