"""Resolve evolving coverage within immutable component computational support.

Masks are dimensionless grid-shaped runtime fields. Geometry and remapping
weights remain static; continuous fractions remain traced for first-order AD.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor._numerical_safety import _require_zero, require_mask_values
from vercor.dtypes import DTypePolicy, as_jax_real_array, jax_ones
from vercor.exceptions import CouplerError
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.contracts import Component


def resolve_component_masks(
    component: Component,
    fields: Mapping[str, RuntimeArray],
    dtype: DTypePolicy,
) -> tuple[RuntimeArray, RuntimeArray]:
    """Return current binary support and fractional coverage on the grid.

    Referenced fields must match runtime precision and horizontal shape and be
    zero outside their parent support. Validation is safe for JIT/JVP/VJP;
    binary transitions themselves have no continuous derivative.
    """
    grid = component.grid
    support = (
        jax_ones(grid.shape, dtype)
        if grid.binary_mask is None
        else as_jax_real_array(grid.binary_mask, dtype)
    )
    declaration = component.spec.masks
    if declaration is None:
        return support, support

    def resolve(
        name: str | None, parent: RuntimeArray, *, binary: bool
    ) -> RuntimeArray:
        if name is None:
            return parent
        owner = f"Component '{component.name}' mask field '{name}'"
        if name not in fields:
            raise CouplerError(f"{owner} is missing from runtime fields.")
        value = jnp.asarray(fields[name])
        if value.shape != grid.shape:
            raise CouplerError(
                f"{owner} has shape {value.shape}; expected {grid.shape}."
            )
        if value.dtype != jnp.dtype(dtype.jax_real):
            raise CouplerError(
                f"{owner} has dtype {value.dtype}; expected {jnp.dtype(dtype.jax_real)}."
            )
        require_mask_values(value, owner=owner, binary=binary)
        _require_zero(
            jnp.count_nonzero((jnp.asarray(parent) == 0) & (value != 0)),
            owner,
            "mask support within the parent binary support",
        )
        return value

    binary_mask = resolve(declaration.binary_field, support, binary=True)
    fraction = resolve(declaration.fractional_field, binary_mask, binary=False)
    return binary_mask, fraction
