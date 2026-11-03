"""Select output names before providers touch array values."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

import jax.numpy as jnp

from vercor.output import OutputContext, JaxOutputContext, OutputFrame, OutputVariable
from vercor.output._dataset import grid_field_dims


def selected_output_names(
    available: Iterable[str], selected: Sequence[str]
) -> tuple[str, ...]:
    """Return an ordered selection, validating unknown names without array work."""
    available = tuple(available)
    names = tuple(selected) or available
    available_set = set(available)
    missing = next((name for name in names if name not in available_set), None)
    if missing is not None:
        raise KeyError(
            f"unknown output variable {missing!r}; available: {', '.join(available) or '<none>'}"
        )
    return names


def component_output_names(
    context: OutputContext | JaxOutputContext,
    native_names: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition selected native/canonical names before either is extracted."""
    native_names = tuple(native_names)
    native_set = set(native_names)
    runtime_names = () if context.state is None else tuple(context.state.fields())
    selected = selected_output_names(
        dict.fromkeys((*native_names, *runtime_names)), context.variables
    )
    return (
        tuple(name for name in selected if name in native_set),
        tuple(name for name in selected if name not in native_set),
    )


def with_runtime_fields(
    frame: OutputFrame,
    context: OutputContext | JaxOutputContext,
    names: Sequence[str],
    *,
    horizontal_dims: tuple[str, str],
) -> OutputFrame:
    """Add selected canonical-only fields in their own component grid order.

    Native entries retain their dimensions and temporal meaning. Canonical fields
    carry the existing coupling-step means, with a singleton sample axis only
    when the provider uses one. No array is extracted unless its name is selected.
    """
    variables = dict(frame.variables)
    coordinates = dict(frame.coordinates)
    for name in names:
        values = context.state.field(name, scope="state")
        grid = context.state.grid
        dims = grid_field_dims(
            name, tuple(values.shape), None if grid is None else grid.shape
        )
        dims = tuple(
            (
                horizontal_dims[0]
                if dim == "nlat"
                else horizontal_dims[1] if dim == "nlon" else dim
            )
            for dim in dims
        )
        if frame.sample_dimension is not None:
            dims = (frame.sample_dimension, *dims)
            values = values[jnp.newaxis, ...]
        variables[name] = OutputVariable(
            dims, values, {"component": context.state.name, "field_name": name}
        )
        if grid is not None:
            for dim, values in zip(
                horizontal_dims, (grid.latitude, grid.longitude), strict=True
            ):
                if dim in dims and dim not in coordinates:
                    coordinates[dim] = OutputVariable((dim,), values)
    if context.variables:
        variables = {name: variables[name] for name in context.variables}
    return replace(frame, variables=variables, coordinates=coordinates)
