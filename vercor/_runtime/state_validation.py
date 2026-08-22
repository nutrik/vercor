from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

from vercor._numerical_safety import require_active_finite
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor._runtime.contracts import ExchangeContract
from vercor.field_layout import validate_component_data_layout
from vercor.fields import _flatten_field_items
from vercor.state import RunState
from vercor._runtime.validation import validate_component_runtime_contract_fields

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding


def validate_runtime_state(
    runtime_state: RunState,
    *,
    components: Mapping[str, _ComponentBinding],
    exchanges: Sequence[Exchange],
    regridders: Mapping[str, Any],
    contracts: Mapping[str, ExchangeContract],
    run_order: Sequence[str] | None,
) -> None:
    """Validate that runtime state matches the configured coupler topology."""

    if run_order is None:
        raise CouplerError("Runtime requires a configured component run sequence")

    run_order = tuple(run_order)
    expected_component_names = tuple(components)
    runtime_component_names = runtime_state._component_names
    if runtime_component_names != expected_component_names:
        expected_set = set(expected_component_names)
        runtime_set = set(runtime_component_names)
        missing = sorted(expected_set - runtime_set)
        extra = sorted(runtime_set - expected_set)
        duplicates = sorted(
            name for name in runtime_set if runtime_component_names.count(name) > 1
        )
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("extra " + ", ".join(extra))
        if duplicates:
            details.append("duplicate " + ", ".join(duplicates))
        if not details:
            details.append("component order differs")
        raise CouplerError(
            "Runtime component names and order must exactly match registered "
            f"components {expected_component_names!r}; got {runtime_component_names!r} "
            f"({'; '.join(details)})."
        )

    for cname in run_order:
        if cname not in components:
            raise CouplerError(
                f"Run-sequence component '{cname}' is not registered in coupler"
            )
        if cname not in runtime_component_names:
            raise CouplerError(
                f"Run-sequence component '{cname}' is missing from runtime state"
            )

    for cname, component in components.items():
        component_state = runtime_state._component_state(cname)
        contract = contracts[cname]
        runtime_grid = runtime_state._component_grids[
            runtime_state._component_index(cname)
        ]
        _validate_runtime_grid(runtime_grid, component.grid, component_name=cname)
        _validate_component_store_schemas(
            component_state,
            component=component,
            contract=contract,
        )
        if component.spec.execution == "jax":
            _validate_payload_schema(
                component_state.payload,
                component._payload,
                component_name=cname,
            )
        validate_component_data_layout(
            component_name=component.name,
            grid_shape=component.grid.shape,
            data=component_state.fields.to_mapping(),
        )
        validate_component_runtime_contract_fields(
            component,
            component_state,
            contract,
        )

    expected_mask_names = tuple(exchange.route_id for exchange in exchanges)
    _validate_exact_names(
        runtime_state._fractional_masks.field_names,
        expected_mask_names,
        owner="Runtime fractional masks",
    )

    for exchange in exchanges:
        route_id = exchange.route_id
        if exchange.source not in runtime_component_names:
            raise CouplerError(
                f"Exchange source component '{exchange.source}' is missing from runtime state"
            )
        if exchange.target not in runtime_component_names:
            raise CouplerError(
                f"Exchange destination component '{exchange.target}' is missing from runtime state"
            )
        if route_id not in regridders:
            raise CouplerError(
                "Runtime requires an initialized regridder for exchange "
                f"{exchange.route_id}"
            )

        mask_name = route_id
        if mask_name not in runtime_state._fractional_masks.field_names:
            raise CouplerError(
                "Runtime requires an initialized fractional mask for exchange "
                f"{exchange.route_id}"
            )
        destination_shape = components[exchange.target].grid.shape
        mask = runtime_state._fractional_masks.get(mask_name)
        mask_array = jnp.asarray(mask)
        mask_shape = mask_array.shape
        if mask_shape != destination_shape:
            raise CouplerError(
                "Runtime fractional mask for exchange "
                f"{exchange.route_id} has shape {mask_shape}, expected {destination_shape}"
            )
        expected_dtype = components[exchange.target]._dtype_policy.jax_real
        if mask_array.dtype != expected_dtype:
            raise CouplerError(
                "Runtime fractional mask for exchange "
                f"{exchange.route_id} has dtype {mask_array.dtype}, expected "
                f"{expected_dtype}."
            )
        _validate_concrete_mask_values(
            mask_array,
            owner=f"Runtime fractional mask for exchange {exchange.route_id}",
            binary=False,
        )

    received_masks = {
        (exchange.target, field_name): runtime_state._fractional_mask(exchange.route_id)
        for exchange in exchanges
        for field_name in _flatten_field_items(exchange.fields)
    }
    for component in components.values():
        component_state = runtime_state._component_state(component.name)
        _validate_component_store_finiteness(
            component_state,
            component=component,
            received_masks=received_masks,
        )
    for component in components.values():
        component._validate_runtime_state(
            runtime_state._component_state(component.name),
            contracts[component.name],
        )


def _validate_component_store_schemas(
    component_state: Any,
    *,
    component: Any,
    contract: ExchangeContract,
) -> None:
    """Validate exact runtime store names, grid shapes, and dtypes."""

    expected_fields = tuple(
        dict.fromkeys((*component.field_names, *contract.receives, *contract.sends))
    )
    for store_name, expected_names in (
        ("fields", expected_fields),
        ("received", contract.receives),
        ("sent", contract.sends),
    ):
        store = getattr(component_state, store_name)
        owner = f"Component '{component.name}' runtime {store_name}"
        _validate_exact_names(store.field_names, expected_names, owner=owner)
        for field_name in expected_names:
            value = jnp.asarray(store.get(field_name))
            if store_name == "fields" and field_name in component._data:
                expected_shape = jnp.asarray(component._data[field_name]).shape
                shape_owner = (
                    "grid shape"
                    if expected_shape == component.grid.shape
                    else "prepared field shape"
                )
            else:
                expected_shape = component.grid.shape
                shape_owner = "grid shape"
            if value.shape != expected_shape:
                raise CouplerError(
                    f"{owner} field '{field_name}' has shape {value.shape}, "
                    f"expected {shape_owner} {expected_shape}."
                )
            expected_dtype = component._dtype_policy.jax_real
            if value.dtype != expected_dtype:
                raise CouplerError(
                    f"{owner} field '{field_name}' has dtype {value.dtype}, "
                    f"expected {expected_dtype}."
                )


def _validate_component_store_finiteness(
    component_state: Any,
    *,
    component: Any,
    received_masks: Mapping[tuple[str, str], Any],
) -> None:
    """Reject non-finite values before a runtime store crosses its next owner."""

    for store_name in ("fields", "received", "sent"):
        store = getattr(component_state, store_name)
        for field_name in store.field_names:
            inbound_mask = received_masks.get((component.name, field_name))
            active_mask = (
                inbound_mask
                if store_name in {"fields", "received"} and inbound_mask is not None
                else component.grid.binary_mask
            )
            require_active_finite(
                store.get(field_name),
                active_mask=active_mask,
                owner=(
                    f"Component '{component.name}' runtime {store_name} "
                    f"field '{field_name}'"
                ),
            )


def _validate_exact_names(
    actual: Sequence[str],
    expected: Sequence[str],
    *,
    owner: str,
) -> None:
    """Reject missing, extra, duplicate, or reordered runtime names."""

    actual_names = tuple(actual)
    expected_names = tuple(expected)
    if actual_names != expected_names:
        raise CouplerError(
            f"{owner} names must exactly match {expected_names!r}; got "
            f"{actual_names!r}."
        )


def _validate_payload_schema(
    actual: Any,
    expected: Any,
    *,
    component_name: str,
) -> None:
    """Validate one differentiable component payload's static PyTree schema."""

    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    owner = f"Component '{component_name}' runtime payload"
    if cast(Any, actual_tree) != cast(Any, expected_tree):
        raise CouplerError(
            f"{owner} PyTree structure does not match the prepared payload."
        )

    for index, (actual_leaf, expected_leaf) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True)
    ):
        actual_metadata = _array_leaf_metadata(actual_leaf)
        expected_metadata = _array_leaf_metadata(expected_leaf)
        if actual_metadata is None or expected_metadata is None:
            if type(actual_leaf) is not type(expected_leaf):
                raise CouplerError(
                    f"{owner} leaf {index} has type {type(actual_leaf).__name__}, "
                    f"expected {type(expected_leaf).__name__}."
                )
            continue
        actual_shape, actual_dtype = actual_metadata
        expected_shape, expected_dtype = expected_metadata
        if actual_shape != expected_shape:
            raise CouplerError(
                f"{owner} leaf {index} has shape {actual_shape}, expected "
                f"{expected_shape}."
            )
        if actual_dtype != expected_dtype:
            raise CouplerError(
                f"{owner} leaf {index} has dtype {actual_dtype}, expected "
                f"{expected_dtype}."
            )


def _array_leaf_metadata(value: Any) -> tuple[tuple[int, ...], Any] | None:
    """Return array shape/dtype metadata when a payload leaf is numeric."""

    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return None
    return array.shape, array.dtype


def _validate_runtime_grid(
    runtime_grid: Any,
    expected_grid: Any,
    *,
    component_name: str,
) -> None:
    """Validate one state's private grid metadata against preparation."""

    if runtime_grid is None:
        raise CouplerError(f"Component '{component_name}' runtime grid is missing.")
    if type(runtime_grid) is not type(expected_grid):
        raise CouplerError(
            f"Component '{component_name}' runtime grid has type "
            f"{type(runtime_grid).__name__}, expected {type(expected_grid).__name__}."
        )
    if runtime_grid.name != expected_grid.name:
        raise CouplerError(
            f"Component '{component_name}' runtime grid name must be "
            f"{expected_grid.name!r}; got {runtime_grid.name!r}."
        )
    for coordinate_name in (
        "longitude",
        "latitude",
        "longitude_edges",
        "latitude_edges",
    ):
        _validate_grid_array(
            getattr(runtime_grid, coordinate_name),
            getattr(expected_grid, coordinate_name),
            owner=(
                f"Component '{component_name}' runtime grid coordinate "
                f"'{coordinate_name}'"
            ),
            compare_values=True,
            binary=False,
        )
    _validate_grid_array(
        runtime_grid.binary_mask,
        expected_grid.binary_mask,
        owner=f"Component '{component_name}' runtime grid binary mask",
        compare_values=True,
        binary=True,
    )


def _validate_grid_array(
    value: Any | None,
    expected: Any | None,
    *,
    owner: str,
    compare_values: bool,
    binary: bool,
) -> None:
    """Validate presence, shape, dtype, and concrete values of grid metadata."""

    if (value is None) != (expected is None):
        raise CouplerError(f"{owner} presence does not match prepared grid.")
    if value is None or expected is None:
        return
    value_array = jnp.asarray(value)
    expected_array = jnp.asarray(expected)
    if value_array.shape != expected_array.shape:
        raise CouplerError(
            f"{owner} has shape {value_array.shape}, expected {expected_array.shape}."
        )
    if value_array.dtype != expected_array.dtype:
        raise CouplerError(
            f"{owner} has dtype {value_array.dtype}, expected {expected_array.dtype}."
        )
    if binary:
        _validate_concrete_mask_values(value_array, owner=owner, binary=True)
    if compare_values:
        values_equal = jnp.array_equal(value_array, expected_array)
        _require_true(
            values_equal,
            f"{owner} values do not match prepared coordinates.",
        )


def _validate_concrete_mask_values(
    mask: Any,
    *,
    owner: str,
    binary: bool,
) -> None:
    """Validate concrete mask finiteness and binary/fractional range."""

    all_finite = jnp.all(jnp.isfinite(mask))
    _require_true(all_finite, f"{owner} must contain only finite values.")
    if binary:
        valid = jnp.logical_or(mask == 0, mask == 1)
        range_text = "values in {0, 1}"
    else:
        valid = jnp.logical_and(mask >= 0, mask <= 1)
        range_text = "values in [0, 1]"
    all_valid = jnp.all(valid)
    _require_true(all_valid, f"{owner} must contain only {range_text}.")


def _require_true(predicate: Any, message: str) -> None:
    """Raise for a false scalar predicate, including inside JAX transforms."""

    if _array_is_traced(predicate):
        jax.debug.callback(
            lambda concrete: _raise_if_false(concrete, message),
            predicate,
        )
        return
    _raise_if_false(predicate, message)


def _raise_if_false(predicate: Any, message: str) -> None:
    """Raise a stable public validation error for one concrete predicate."""

    if not bool(predicate):
        raise CouplerError(message)


def _array_is_traced(value: Any) -> bool:
    """Return whether a value is represented by a JAX tracer."""

    return isinstance(value, jax.core.Tracer)
