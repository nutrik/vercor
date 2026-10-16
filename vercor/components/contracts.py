"""Public protocol-first component authoring contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from inspect import signature
from typing import Any, Literal, TypeAlias

import jax

from vercor._field_names import (
    freeze_name_sequence as _freeze_name_sequence,
    unique_field_names as _unique_field_names,
)
import vercor.components._protocol as _component_protocol
from vercor.components._protocol import (
    _snapshot_mapping,
    Component,
    StepResult,
)
from vercor.components.contexts import SetupContext, StepContext
from vercor.exceptions import ComponentError
from vercor.state import ComponentState
from vercor.types import RuntimeArray


def _validate_callback(name: str, callback: object | None) -> None:
    """Validate one optional lifecycle callback immediately."""

    if callback is None:
        return
    if not callable(callback):
        raise TypeError(f"LifecycleHooks.{name} must be callable or None")
    try:
        callback_signature = signature(callback)
        callback_signature.bind(object(), object())
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"LifecycleHooks.{name} must accept exactly (component, context)"
        ) from exc
    try:
        callback_signature.bind(object(), object(), object())
    except TypeError:
        return
    raise TypeError(f"LifecycleHooks.{name} must accept exactly (component, context)")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class SetupResult:
    """Return setup-owned field and payload state from a setup callback.

    ``fields`` may contain declared scalar or canonical grid-layout author
    values. The private preparation boundary expands scalars, applies the
    runtime dtype, validates layouts, and copy-owns mutable NumPy leaves.
    ``payload`` may be any JAX PyTree. Preparation and state creation rebuild
    its standard PyTree containers, copies NumPy leaves, and deep-copies opaque
    object leaves (or rejects leaves that cannot be copied). The result becomes
    the initial per-component runtime payload.
    """

    fields: Mapping[str, object]
    payload: Any | None

    def __init__(
        self,
        fields: Mapping[str, object] | None = None,
        payload: Any | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "fields",
            _snapshot_mapping(fields, label="SetupResult.fields"),
        )
        object.__setattr__(self, "payload", payload)

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        """Flatten fields and payload while preserving mapping order."""

        return (*self.fields.values(), self.payload), tuple(self.fields)

    @classmethod
    def tree_unflatten(
        cls, names: tuple[str, ...], children: tuple[Any, ...]
    ) -> "SetupResult":
        """Restore a setup result from JAX PyTree leaves."""

        field_count = len(names)
        return cls(
            dict(zip(names, children[:field_count], strict=True)),
            children[field_count],
        )


@dataclass(frozen=True, init=False)
class PrefillContext:
    """Read-only runtime stores and exchange declarations for prefill.

    The callback receives snapshots of component ``fields`` plus ``received``
    and ``sent`` exchange stores. ``receives`` and ``sends`` are the exact
    coupler-owned contract names that a :class:`PrefillResult` may populate.
    """

    fields: Mapping[str, RuntimeArray]
    received: Mapping[str, RuntimeArray]
    sent: Mapping[str, RuntimeArray]
    receives: tuple[str, ...]
    sends: tuple[str, ...]

    def __init__(
        self,
        fields: Mapping[str, RuntimeArray],
        received: Mapping[str, RuntimeArray],
        sent: Mapping[str, RuntimeArray],
        receives: Iterable[str] = (),
        sends: Iterable[str] = (),
    ) -> None:
        object.__setattr__(self, "fields", _snapshot_mapping(fields))
        object.__setattr__(self, "received", _snapshot_mapping(received))
        object.__setattr__(self, "sent", _snapshot_mapping(sent))
        object.__setattr__(self, "receives", _unique_field_names(receives))
        object.__setattr__(self, "sends", _unique_field_names(sends))


@dataclass(frozen=True, init=False)
class PrefillResult:
    """Return normalized-on-application updates from a prefill callback.

    ``fields`` must be declared component inputs or outputs. ``received`` and
    ``sent`` must be named by the active exchange contract and normalize to the
    exact component-grid shape. All three mappings are immutable snapshots;
    scalar values expand and values adopt the prepared runtime dtype before any
    store is updated.
    """

    fields: Mapping[str, object]
    received: Mapping[str, object]
    sent: Mapping[str, object]

    def __init__(
        self,
        fields: Mapping[str, object] | None = None,
        received: Mapping[str, object] | None = None,
        sent: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(self, "fields", _snapshot_mapping(fields))
        object.__setattr__(self, "received", _snapshot_mapping(received))
        object.__setattr__(self, "sent", _snapshot_mapping(sent))


@dataclass(frozen=True)
class ValidationContext:
    """Public runtime state supplied to a validation callback."""

    state: ComponentState
    payload: Any | None = None
    receives: tuple[str, ...] = ()
    sends: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze exchange-name sequences without splitting text scalars."""

        object.__setattr__(
            self,
            "receives",
            _freeze_name_sequence(
                self.receives,
                label="ValidationContext.receives",
            ),
        )
        object.__setattr__(
            self,
            "sends",
            _freeze_name_sequence(
                self.sends,
                label="ValidationContext.sends",
            ),
        )


_ComponentStepReturn: TypeAlias = Mapping[str, RuntimeArray] | StepResult
_ComponentStepCallable: TypeAlias = Callable[
    [Mapping[str, RuntimeArray], StepContext, Any | None], _ComponentStepReturn
]
_AuthorStepCallable: TypeAlias = Callable[..., _ComponentStepReturn]


@dataclass(frozen=True)
class LifecycleHooks:
    """Optional callbacks at the three component lifecycle boundaries.

    ``setup(owner, context)`` runs once during private binding preparation and
    returns :class:`SetupResult` or ``None``. ``prefill(owner, context)`` runs
    when runtime stores are initially created and returns
    :class:`PrefillResult` or ``None``. ``validate(owner, context)`` runs during
    runtime-state validation and returns ``None``. Every callback receives the
    original author object, never the private binding.
    """

    setup: Callable[[Component, SetupContext], SetupResult | None] | None = None
    prefill: Callable[[Component, PrefillContext], PrefillResult | None] | None = None
    validate: Callable[[Component, ValidationContext], None] | None = None

    def __post_init__(self) -> None:
        """Reject invalid nested callbacks at configuration time."""

        _validate_callback("setup", self.setup)
        _validate_callback("prefill", self.prefill)
        _validate_callback("validate", self.validate)


@dataclass(frozen=True)
class TransferPolicy:
    """Select how time-dependent component data is exported.

    ``current`` sends the stored field directly, ``linear`` interpolates the
    adjacent monthly samples using runtime weights, and ``daily`` selects the
    active daily sample. The mode is static component policy, not traced
    physics.
    """

    time_selection: Literal["current", "linear", "daily"] = "current"

    def __post_init__(self) -> None:
        """Validate the explicit time-selection mode."""

        if self.time_selection not in ("current", "linear", "daily"):
            raise ValueError("time_selection must be 'current', 'linear', or 'daily'")


@dataclass(frozen=True)
class ComponentMaskSpec:
    """Reference evolving binary/fractional fields on a component's static grid.

    Both fields use the current grid-shaped runtime layout. Omitted binary
    support defaults to the grid wet mask; omitted fractional coverage is one
    on the binary support. Fractional fields remain differentiable inputs.
    """

    binary_field: str | None = None
    fractional_field: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed field references before component preparation."""
        for label in ("binary_field", "fractional_field"):
            value = getattr(self, label)
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError(f"{label} must be a field name or None")
                if not value.strip():
                    raise ValueError(f"{label} must be a non-empty field name")


@dataclass(frozen=True, init=False)
class ComponentSpec:
    """Declare all author-controlled component runtime capabilities.

    ``inputs`` are readable fields and ``outputs`` are writable step results;
    both are validated, deduplicated name tuples. ``initial_fields`` is a
    defensive immutable snapshot containing only declared names. Scalar values
    expand on the grid and all values adopt the runtime dtype during private
    preparation. ``execution`` selects differentiable JAX capability or the
    Python host path. ``lifecycle``, ``transfer``, and ``output`` own the sole
    setup/runtime-hook, time-selection, and output policies respectively.
    ``masks`` optionally references current binary/fractional runtime fields;
    computational grid support stays immutable.
    """

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    initial_fields: Mapping[str, object]
    execution: Literal["jax", "host"]
    lifecycle: LifecycleHooks
    transfer: TransferPolicy
    output: "OutputSpec"
    masks: ComponentMaskSpec | None

    def __init__(
        self,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        initial_fields: Mapping[str, object] | None = None,
        *,
        execution: Literal["jax", "host"] = "jax",
        lifecycle: LifecycleHooks | None = None,
        transfer: TransferPolicy | None = None,
        output: "OutputSpec | None" = None,
        masks: ComponentMaskSpec | None = None,
    ) -> None:
        """Create and eagerly validate a component declaration."""

        normalized_inputs = _unique_field_names(inputs)
        normalized_outputs = _unique_field_names(outputs)
        frozen_initial_fields = _snapshot_mapping(
            initial_fields,
            label="ComponentSpec.initial_fields",
        )
        declared = set((*normalized_inputs, *normalized_outputs))
        undeclared = next(
            (name for name in frozen_initial_fields if name not in declared), None
        )
        if undeclared is not None:
            raise ComponentError(
                f"ComponentSpec initial field '{undeclared}' is not declared in "
                "inputs or outputs."
            )
        if execution not in ("jax", "host"):
            raise ValueError("execution must be 'jax' or 'host'")
        if lifecycle is not None and not isinstance(lifecycle, LifecycleHooks):
            raise TypeError("lifecycle must be LifecycleHooks or None")
        if transfer is not None and not isinstance(transfer, TransferPolicy):
            raise TypeError("transfer must be TransferPolicy or None")
        if output is not None and not isinstance(output, OutputSpec):
            raise TypeError("output must be OutputSpec or None")
        if masks is not None:
            if not isinstance(masks, ComponentMaskSpec):
                raise TypeError("masks must be ComponentMaskSpec or None")
            for name in (masks.binary_field, masks.fractional_field):
                if name is not None and name not in declared:
                    raise ComponentError(
                        f"Component mask field '{name}' must be declared."
                    )
            if transfer is not None and transfer.time_selection != "current":
                raise ComponentError("Component masks require current time selection.")
        object.__setattr__(self, "inputs", normalized_inputs)
        object.__setattr__(self, "outputs", normalized_outputs)
        object.__setattr__(self, "initial_fields", frozen_initial_fields)
        object.__setattr__(self, "execution", execution)
        object.__setattr__(
            self, "lifecycle", LifecycleHooks() if lifecycle is None else lifecycle
        )
        object.__setattr__(
            self, "transfer", TransferPolicy() if transfer is None else transfer
        )
        object.__setattr__(self, "output", OutputSpec() if output is None else output)
        object.__setattr__(self, "masks", masks)


# This late public-owner import keeps component and output declarations acyclic
# while retaining one canonical owner for each public contract.
from vercor.output import OutputSpec  # noqa: E402

# Resolve the protocol's public forward reference once both declarations exist.
# This is import metadata, not runtime component state.
_component_protocol._resolve_component_spec(ComponentSpec)
_KEEP_PAYLOAD = _component_protocol._KEEP_PAYLOAD

__all__ = [
    "Component",
    "ComponentMaskSpec",
    "ComponentSpec",
    "LifecycleHooks",
    "PrefillContext",
    "PrefillResult",
    "SetupContext",
    "SetupResult",
    "StepContext",
    "StepResult",
    "TransferPolicy",
    "ValidationContext",
]
