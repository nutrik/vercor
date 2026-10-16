from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime
from importlib import import_module
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    Component,
    ComponentSpec,
    LifecycleHooks,
    SetupResult,
)
from vercor.components.contexts import SetupContext, StepContext
from vercor.coupler import Coupler
from vercor.dtypes import DTypePolicy, dtype_policy
from vercor.fluxes.utilities import compute_air_density
from vercor.jax_logging import get_default_logger
from vercor.runtime import RuntimeOptions

EXPECTED_CONSTANTS = {
    "earth_radius": 6.371e6,
    "gravity": 9.81,
    "air_density": 1.3,
    "dry_air_gas_constant": 287.042,
    "dry_air_specific_heat": 1.00464e3,
    "water_vapor_mass_ratio_correction": 0.608,
    "reference_pressure": 1e5,
    "dry_air_molecular_weight": 28.966,
    "water_vapor_specific_heat": 1.810e3,
    "water_vapor_specific_heat_ratio_correction": 0.802,
    "dry_air_kappa": 0.286,
    "ice_latent_heat_of_fusion": 3.337e5,
    "universal_gas_constant": 8314.47,
    "ocean_minimum_wind_speed": 0.5,
    "ice_minimum_wind_speed": 1.0,
    "von_karman_constant": 0.4,
    "stefan_boltzmann_constant": 5.67e-8,
    "ocean_emissivity": 0.97,
    "ice_emissivity": 0.97,
    "snow_emissivity": 0.99,
    "latent_heat_of_vaporization": 2.501e6,
    "freshwater_latent_heat_of_fusion": 3.34e5,
    "bulk_aerodynamic_resistance": 0.1,
    "reference_height": 10.0,
    "air_temperature_reference_height": 2.0,
    "freshwater_density": 1000.0,
    "surface_salinity_reference": 34.7,
    "precipitation_snow_temperature": 272.77,
    "precipitation_rain_temperature": 278.15,
}


def _physical_constants_type() -> type[Any]:
    return cast(type[Any], import_module("vercor.physics").PhysicalConstants)


def test_physical_constants_have_canonical_names_and_preserved_defaults() -> None:
    physical_constants_type = _physical_constants_type()
    constants = physical_constants_type()

    assert tuple(field.name for field in fields(constants)) == tuple(EXPECTED_CONSTANTS)
    assert {
        field.name: getattr(constants, field.name) for field in fields(constants)
    } == EXPECTED_CONSTANTS
    assert not hasattr(constants, "enable_x64")
    assert not hasattr(constants, "dtype")


def test_physical_constants_are_keyword_only() -> None:
    physical_constants_type = _physical_constants_type()

    with pytest.raises(TypeError, match="positional"):
        physical_constants_type(6.4e6)


def test_physical_constants_document_every_field_and_ambiguous_units() -> None:
    physical_constants_type = _physical_constants_type()
    docstring = physical_constants_type.__doc__ or ""

    for name in EXPECTED_CONSTANTS:
        assert f"{name}:" in docstring
    assert "dry_air_gas_constant: Specific gas constant" in docstring
    assert "J/(kg K)" in docstring
    assert "dry_air_molecular_weight: Molecular weight" in docstring
    assert "kg/kmol" in docstring
    assert "universal_gas_constant: Universal molar gas constant" in docstring
    assert "J/(kmol K)" in docstring
    assert "ice_latent_heat_of_fusion: Sea-ice" in docstring
    assert "freshwater_latent_heat_of_fusion: Freshwater" in docstring


def test_physical_constants_are_a_frozen_registered_pytree() -> None:
    physical_constants_type = _physical_constants_type()
    constants = physical_constants_type()

    leaves, tree_definition = jax.tree_util.tree_flatten(constants)
    restored = jax.tree_util.tree_unflatten(tree_definition, leaves)

    assert len(leaves) == len(EXPECTED_CONSTANTS)
    assert restored == constants
    with pytest.raises(FrozenInstanceError):
        constants.gravity = 10.0


def test_physical_constants_normalize_mutable_numpy_scalar_inputs() -> None:
    physical_constants_type = _physical_constants_type()
    source = np.asarray(9.81)

    constants = physical_constants_type(gravity=source)
    source[...] = 1.0

    assert isinstance(constants.gravity, float)
    assert constants.gravity == 9.81


@pytest.mark.parametrize(
    "value",
    [np.asarray([9.81]), jnp.asarray([9.81])],
    ids=("numpy", "jax"),
)
def test_physical_constants_reject_non_scalar_leaves(value: Any) -> None:
    physical_constants_type = _physical_constants_type()

    with pytest.raises(TypeError, match="gravity.*scalar"):
        physical_constants_type(gravity=value)


@pytest.mark.parametrize(
    "value",
    [
        np.asarray("9.81"),
        np.asarray(True),
        np.asarray(object(), dtype=object),
        jnp.asarray(True),
    ],
    ids=("numpy-string", "numpy-bool", "numpy-object", "jax-bool"),
)
def test_physical_constants_reject_nonnumeric_scalar_leaves(value: Any) -> None:
    physical_constants_type = _physical_constants_type()

    with pytest.raises(TypeError, match="gravity.*numeric scalar"):
        physical_constants_type(gravity=value)


def test_physical_constants_support_forward_and_reverse_gradients() -> None:
    physical_constants_type = _physical_constants_type()
    constants = physical_constants_type()
    tangent = physical_constants_type(**{name: 1.0 for name in EXPECTED_CONSTANTS})

    def objective(values: Any) -> Any:
        return values.gravity * values.air_density

    value, forward_tangent = jax.jvp(objective, (constants,), (tangent,))
    reverse_gradient = jax.grad(objective)(constants)

    assert_allclose_compact(value, np.asarray(9.81 * 1.3))
    assert_allclose_compact(forward_tangent, np.asarray(9.81 + 1.3))
    assert_allclose_compact(reverse_gradient.gravity, np.asarray(1.3))
    assert_allclose_compact(reverse_gradient.air_density, np.asarray(9.81))
    assert_allclose_compact(reverse_gradient.earth_radius, np.asarray(0.0))


def test_flux_consumes_canonical_constants_and_differentiates_through_them() -> None:
    physical_constants_type = _physical_constants_type()
    constants = physical_constants_type()
    pressure = jnp.asarray([100000.0, 85000.0])
    temperature = jnp.asarray([280.0, 260.0])

    density = compute_air_density(constants, pressure, temperature)

    expected = (
        constants.dry_air_molecular_weight
        / constants.universal_gas_constant
        * pressure
        / temperature
    )
    assert_allclose_compact(density, expected)

    def loss(molecular_weight: Any) -> Any:
        configured = replace(
            constants,
            dry_air_molecular_weight=molecular_weight,
        )
        return jnp.sum(compute_air_density(configured, pressure, temperature))

    forward = jax.jvp(loss, (jnp.asarray(28.0),), (jnp.asarray(1.0),))[1]
    reverse = jax.grad(loss)(jnp.asarray(28.0))

    assert_allclose_compact(forward, reverse)
    assert bool(jnp.isfinite(reverse))
    assert float(reverse) > 0.0


def test_dtype_helpers_reject_settings_as_a_precision_owner() -> None:
    physical_constants_type = _physical_constants_type()

    assert not hasattr(DTypePolicy, "from_settings")
    with pytest.raises(TypeError, match="DTypePolicy"):
        dtype_policy(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        physical_constants_type(dtype=DTypePolicy(enable_x64=True))


def test_runtime_options_reject_settings_as_dtype_immediately() -> None:
    with pytest.raises(TypeError, match="dtype.*DTypePolicy"):
        RuntimeOptions(dtype=object())  # type: ignore[arg-type]


def test_setup_and_step_contexts_expose_constants() -> None:
    physical_constants_type = _physical_constants_type()
    constants = physical_constants_type(gravity=10.0)

    setup = SetupContext(
        start=datetime(2000, 1, 1),
        dt_seconds=60.0,
        run_order=("MODEL",),
        constants=constants,
        logger=get_default_logger(),
    )
    step = StepContext(
        dt_seconds=60.0,
        constants=constants,
    )

    assert setup.constants is constants
    assert step.constants is constants


@pytest.mark.fast_always
def test_coupler_wires_constants_while_runtime_options_own_dtype() -> None:
    physical_constants_type = _physical_constants_type()
    jax.config.update("jax_enable_x64", True)
    constants = physical_constants_type(gravity=jnp.asarray(3.0, dtype=jnp.float64))
    observed_setup_constants: list[Any] = []

    def setup(component: Component, context: SetupContext) -> SetupResult:
        _ = component
        observed_setup_constants.append(context.constants)
        return SetupResult(fields={"value": 2.0})

    component = CallableComponent(
        "MODEL",
        make_test_grid(name="physics-contract"),
        lambda fields, context: {"value": fields["value"] * context.constants.gravity},
        spec=ComponentSpec(
            outputs=("value",),
            execution="jax",
            lifecycle=LifecycleHooks(setup=setup),
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=False)),
        constants=constants,
    )
    state = coupler.initial_state()
    final_state = coupler.run(state)
    result = final_state.component("MODEL").field("value")

    assert len(observed_setup_constants) == 1
    assert observed_setup_constants[0].gravity.dtype == jnp.float32
    assert result.dtype == jnp.float32
    assert_allclose_compact(result, np.full(component.grid.shape, 6.0))


@pytest.mark.fast_always
def test_runtime_dtype_cast_preserves_constant_gradient() -> None:
    physical_constants_type = _physical_constants_type()
    jax.config.update("jax_enable_x64", True)

    def loss(gravity: Any) -> Any:
        component = CallableComponent(
            "MODEL",
            make_test_grid(name="physics-gradient"),
            lambda fields, context: {
                "value": fields["value"] * context.constants.gravity
            },
            spec=ComponentSpec(
                outputs=("value",),
                initial_fields={"value": 2.0},
                execution="jax",
            ),
        )
        coupler = Coupler(
            Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            components=(component,),
            run_order=("MODEL",),
            runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=False)),
            constants=physical_constants_type(gravity=gravity),
        )
        final_state = coupler.run(coupler.initial_state())
        result = final_state.component("MODEL").field("value")
        assert result.dtype == jnp.float32
        return jnp.mean(result)

    gradient = jax.grad(loss)(jnp.asarray(3.0, dtype=jnp.float64))

    assert_allclose_compact(gradient, np.asarray(2.0))


@pytest.mark.fast_always
def test_numpy_source_mutation_cannot_stale_prepared_constants() -> None:
    physical_constants_type = _physical_constants_type()
    source = np.asarray(3.0)
    constants = physical_constants_type(gravity=source)
    component = CallableComponent(
        "MODEL",
        make_test_grid(name="numpy-constant"),
        lambda fields, context: {"value": fields["value"]},
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": 2.0},
            execution="jax",
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        constants=constants,
    )

    coupler.initial_state()
    source[...] = 8.0
    coupler.initial_state()

    assert constants.gravity == 3.0


@pytest.mark.fast_always
def test_coupler_constants_are_read_only_after_construction() -> None:
    physical_constants_type = _physical_constants_type()
    component = CallableComponent(
        "MODEL",
        make_test_grid(name="replaced-constants"),
        lambda fields, context: {"value": fields["value"]},
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": 2.0},
            execution="jax",
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        constants=physical_constants_type(),
    )
    with pytest.raises(AttributeError, match="constants.*no setter"):
        coupler.constants = physical_constants_type(gravity=10.0)  # type: ignore[misc]
