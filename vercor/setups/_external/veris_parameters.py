"""Immutable, traced updates of native sea-ice physical parameters.

Native Veris owns parameter names, dependencies and domains. VerCOR owns runtime
precision and freshwater density. This bridge changes only the native physical
PyTree in one initialized component payload; fields and native state are retained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import jax.numpy as jnp

from vercor.exceptions import ComponentError
from vercor.state import RunState


def replace_veris_parameters(
    state: RunState,
    parameters: Mapping[str, object],
    *,
    component: str = "CICE",
) -> RunState:
    """Return a state with updated scalar native physics and fixed initial arrays.

    ``parameters`` contains independent native physical names, such as ``h0``.
    Unknown/derived/numerical names, nonscalar/complex values and direct rhoFresh
    updates are rejected. VerCOR's freshwater_density supplies rhoFresh at each
    transition. Native host validation checks concrete domains; compiled advance
    returns explicit validity status for traced values. Hard ice transitions keep
    their native selected-branch derivatives. Restart each calibration candidate
    from the same initial state and hold geometry/precision fixed.
    """
    from vercor.setups._external.veris_runtime import (
        VerisPayload,
        _runtime_veris_constants,
    )

    if not isinstance(state, RunState):
        raise TypeError("state must be RunState")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping of scalar physical values")
    current = state._component_state(component)
    payload = current.payload
    if not isinstance(payload, VerisPayload):
        raise ComponentError(
            f"Component '{component}' requires initialized standalone Veris physics"
        )
    if not all(isinstance(name, str) for name in parameters):
        raise TypeError("physical parameter names must be strings")
    if "rhoFresh" in parameters:
        raise ValueError("rhoFresh is owned by VerCOR freshwater_density")
    allowed = payload.physical_constants.base_parameter_names()
    unknown = set(parameters) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown or non-independent Veris physical parameters: {sorted(unknown)}"
        )
    values = {}
    for name, value in parameters.items():
        array = jnp.asarray(value)
        if array.shape != () or array.dtype.kind not in "fiu":
            raise TypeError(
                f"Veris physical parameter '{name}' must be a real numeric scalar"
            )
        values[name] = array.astype(payload.physical_constants.dtype)
    updated = _runtime_veris_constants(
        payload.physical_constants.with_overrides(**values)
    )
    return state._with_component_state(
        component, current.with_payload(replace(payload, physical_constants=updated))
    )
