"""Validate current component mask fields without changing grid geometry."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vercor.components as api
from vercor import Clock, Coupler, RectilinearGrid, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.exceptions import ComponentError, CouplerError

pytestmark = pytest.mark.fast_always
_DTYPE = DTypePolicy(enable_x64=True)


def _mask_spec(**kwargs: Any) -> Any:
    mask_type = getattr(api, "ComponentMaskSpec", None)
    assert mask_type is not None, "ComponentMaskSpec is missing"
    return mask_type(**kwargs)


def _component(
    *, binary: bool = True, fractional: bool = True, masked: bool = True
) -> api.Component:
    grid = RectilinearGrid(
        "ocean",
        longitude=jnp.array([0.0, 1.0, 2.0]),
        latitude=jnp.array([0.0, 1.0]),
        binary_mask=jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    masks = (
        _mask_spec(
            binary_field="active" if binary else None,
            fractional_field="area" if fractional else None,
        )
        if masked
        else None
    )
    return api.CallableComponent(
        "CICE",
        grid,
        lambda fields: {},
        spec=api.ComponentSpec(
            outputs=("active", "area", "temperature"),
            masks=masks,
            initial_fields={"active": 0.0, "area": 0.0, "temperature": 270.0},
        ),
    )


def _resolve(component: api.Component, fields: dict[str, Any]) -> Any:
    from vercor._runtime.component_masks import resolve_component_masks

    return resolve_component_masks(component, fields, _DTYPE)


@pytest.mark.parametrize("field", ["binary_field", "fractional_field"])
@pytest.mark.parametrize("value", ["", " ", 123])
def test_mask_spec_rejects_invalid_field_names(field: str, value: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="field"):
        _mask_spec(**{field: value})


def test_mask_spec_references_declared_current_fields() -> None:
    with pytest.raises(ComponentError, match="mask.*area.*declared"):
        api.ComponentSpec(
            outputs=("temperature",), masks=_mask_spec(fractional_field="area")
        )
    declaration = api.ComponentSpec(
        outputs=("area",),
        masks=_mask_spec(fractional_field="area"),
        transfer=api.TransferPolicy(time_selection="linear"),
    )
    assert declaration.masks is not None
    assert declaration.masks.fractional_field == "area"
    with pytest.raises(TypeError, match="masks"):
        api.ComponentSpec(masks="area")  # type: ignore[arg-type]


def test_component_masks_respect_static_support() -> None:
    component = _component()
    static = np.asarray(component.grid.binary_mask).copy()
    fields = {
        "active": jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "area": jnp.array([[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    }
    binary, fraction = _resolve(component, fields)
    np.testing.assert_array_equal(binary, [[1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(fraction, [[0.25, 0, 0], [0.5, 0, 0]])
    np.testing.assert_array_equal(component.grid.binary_mask, static)


def test_default_fraction_follows_binary_support() -> None:
    binary, fraction = _resolve(
        _component(fractional=False),
        {"active": jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
    )
    np.testing.assert_array_equal(binary, fraction)
    np.testing.assert_array_equal(fraction, [[1, 0, 0], [1, 0, 0]])


def test_static_component_mask_defaults_are_unchanged() -> None:
    component = _component(masked=False)
    binary, fraction = _resolve(component, {})
    np.testing.assert_array_equal(binary, [[1, 1, 0], [1, 0, 0]])
    np.testing.assert_array_equal(fraction, binary)


@pytest.mark.parametrize(
    "field,value",
    [
        ("active", 0.5),
        ("active", jnp.nan),
        ("active", jnp.inf),
        ("area", -0.1),
        ("area", 1.1),
        ("area", jnp.nan),
        ("area", jnp.inf),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_mask_values_reject_invalid_inputs(
    field: str, value: float, compiled: bool
) -> None:
    component = _component()
    fields = {
        "active": jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "area": jnp.array([[0.25, 0.5, 0.0], [0.75, 0.0, 0.0]]),
    }
    fields[field] = fields[field].at[0, 0].set(value)

    def operation(x: dict[str, Any]) -> Any:
        return _resolve(component, x)

    if compiled:
        operation = jax.jit(operation)
    with pytest.raises(Exception, match=f"CICE.*{field}"):
        jax.block_until_ready(operation(fields))


@pytest.mark.parametrize("field", ["active", "area"])
def test_masks_cannot_activate_outside_their_support(field: str) -> None:
    fields = {"active": jnp.zeros((2, 3)), "area": jnp.zeros((2, 3))}
    fields[field] = fields[field].at[0, 2].set(1.0)
    with pytest.raises(CouplerError, match="CICE.*support"):
        _resolve(_component(), fields)


def test_fraction_cannot_activate_outside_dynamic_binary_support() -> None:
    with pytest.raises(CouplerError, match="CICE.*support"):
        _resolve(_component(), {"active": jnp.zeros((2, 3)), "area": jnp.ones((2, 3))})


@pytest.mark.parametrize(
    "value", [jnp.zeros((1, 3)), jnp.zeros((2, 3), dtype=jnp.float32)]
)
def test_mask_shape_and_dtype_are_fixed(value: jax.Array) -> None:
    with pytest.raises(CouplerError, match="CICE.*area.*(shape|dtype)"):
        _resolve(_component(binary=False), {"area": value})


def test_mask_field_is_required_in_runtime_state() -> None:
    with pytest.raises(CouplerError, match="CICE.*area"):
        _resolve(_component(binary=False), {})


def test_fractional_mask_remains_a_differentiable_runtime_leaf() -> None:
    component = _component(binary=False)
    support = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    def objective(x: jax.Array) -> jax.Array:
        return jnp.sum(_resolve(component, {"area": x * support})[1])

    value, tangent = jax.jvp(jax.jit(objective), (jnp.array(0.25),), (jnp.array(1.0),))
    np.testing.assert_allclose(value, 0.75)
    np.testing.assert_allclose(tangent, 3.0)
    np.testing.assert_allclose(jax.grad(objective)(jnp.array(0.25)), 3.0)


def test_setup_validates_masks_after_scalar_expansion() -> None:
    component = _component(binary=False)
    bad = api.CallableComponent(
        component.name,
        component.grid,
        lambda fields: {},
        spec=api.ComponentSpec(
            outputs=("area",),
            initial_fields={"area": 1.1},
            masks=_mask_spec(fractional_field="area"),
        ),
    )
    with pytest.raises(CouplerError, match="CICE.*area"):
        Coupler(
            Clock(datetime(2000, 1, 1), 60.0, 1),
            components=(bad,),
            run_order=("CICE",),
            runtime=RuntimeOptions(dtype=_DTYPE),
        ).initial_state()
