"""Public native-physics updates are immutable, traced and precision-owned."""

from dataclasses import FrozenInstanceError
from datetime import timedelta
import json
import subprocess
import sys
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_veris_component import small_coupler
from tests.test_veris_rollouts import native_payload
from vercor.exceptions import CouplerError
from vercor.setups import VerisConfig, replace_veris_parameters

pytestmark = pytest.mark.slow


def configured(**kwargs: Any) -> VerisConfig:
    return VerisConfig(
        model_timestep=timedelta(seconds=300),
        custom_parameters={"nEVPsteps": 4},
        initial_conditions={"Area": 0.5, "hIceMean": 0.25, "TSurf": 260.0},
        **kwargs,
    )


@pytest.mark.fast_always
def test_public_parameter_replacement_differentiates_without_mutation() -> None:
    coupler, _, _, _ = small_coupler(config=configured())
    initial = coupler.initial_state()
    original = [np.asarray(x).copy() for x in jax.tree.leaves(initial)]
    original_value = np.asarray(
        coupler.run(initial).component("CICE").field("ice_fraction")
    )

    @jax.jit
    def concentration_loss(value: jax.Array) -> jax.Array:
        candidate = replace_veris_parameters(initial, {"h0": value})
        final = coupler.run(candidate)
        return jnp.mean(final.component("CICE").field("ice_fraction"))

    point = jnp.asarray(0.43)
    forward = jax.jacfwd(concentration_loss)(point)
    reverse = jax.grad(concentration_loss)(point)
    assert np.isfinite(float(reverse)) and abs(float(reverse)) > 1e-10
    np.testing.assert_allclose(forward, reverse, rtol=1e-10)
    errors = []
    for relative in (1e-2, 1e-3, 1e-4):
        step = point * relative
        fd = (concentration_loss(point + step) - concentration_loss(point - step)) / (
            2 * step
        )
        errors.append(abs(float((fd - reverse) / reverse)))
    assert max(errors[1:]) < 2e-4, f"ERROR h0 FD plateau:{errors}"
    for value in (0.4, 0.6, 0.4):
        candidate = replace_veris_parameters(initial, {"h0": value})
        assert candidate is not initial
        assert native_payload(candidate).state is native_payload(initial).state
        assert float(native_payload(candidate).physical_constants.h0) == value
        concentration_loss(jnp.asarray(value))
    for actual, expected in zip(jax.tree.leaves(initial), original, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        coupler.run(initial).component("CICE").field("ice_fraction"), original_value
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"unknown": 0.5},
        {"recip_h0": 2.0},
        {"nEVPsteps": 4},
        {"dtype": "float32"},
        {"h0": np.array([0.43])},
        {"h0": 1 + 2j},
        {"h0": True},
        {"rhoFresh": 1000.0},
        {"h0": -1.0},
        {"h0": float("nan")},
    ],
)
def test_rejects_invalid_or_owned_parameter_updates(
    parameters: dict[str, object],
) -> None:
    coupler, _, _, _ = small_coupler(config=configured())
    initial = coupler.initial_state()
    with pytest.raises((ValueError, TypeError, CouplerError)):
        replace_veris_parameters(initial, parameters)


@pytest.mark.parametrize("component", ["MISSING", "FORCING"])
def test_rejects_absent_or_non_veris_component(component: str) -> None:
    coupler, _, _, _ = small_coupler(config=configured())
    with pytest.raises((KeyError, ValueError, TypeError, CouplerError)):
        replace_veris_parameters(
            coupler.initial_state(), {"h0": 0.43}, component=component
        )


@pytest.mark.parametrize("enable_x64", [False, True])
def test_parameter_leaves_follow_runtime_precision(enable_x64: bool) -> None:
    coupler, _, _, _ = small_coupler(
        config=configured(physical_constants={"h0": 0.43}), enable_x64=enable_x64
    )
    initial = coupler.initial_state()
    candidate = replace_veris_parameters(initial, {"h0": jnp.asarray(0.6)})
    expected = jnp.float64 if enable_x64 else jnp.float32
    assert all(
        leaf.dtype == expected
        for leaf in jax.tree.leaves(native_payload(candidate).physical_constants)
    )
    np.testing.assert_allclose(
        native_payload(candidate).physical_constants.recip_h0, 1 / 0.6, rtol=1e-6
    )
    assert all(
        x.dtype == expected
        for x in jax.tree.leaves(native_payload(coupler.run(candidate)))
    )


@pytest.mark.fast_always
def test_configuration_owns_physical_parameter_mapping_and_arrays() -> None:
    value = np.asarray(0.43)
    supplied: dict[str, object] = {"h0": value}
    config = configured(physical_constants=supplied)
    supplied["h0"] = 0.9
    value[...] = 0.7
    assert config.physical_constants is not None
    assert float(cast(Any, config.physical_constants["h0"])) == 0.43
    with pytest.raises(TypeError):
        cast(Any, config.physical_constants)["h0"] = 0.8
    with pytest.raises(ValueError):
        cast(Any, config.physical_constants["h0"])[...] = 0.8
    with pytest.raises(FrozenInstanceError):
        setattr(config, "physical_constants", {})


@pytest.mark.fast_always
def test_parameter_helper_export_does_not_import_optional_models() -> None:
    code = """
import builtins, json, sys
original=builtins.__import__
def block(name,*args,**kwargs):
    if name.split('.')[0] in {'veris','veros'}:
        raise ImportError('optional model blocked')
    return original(name,*args,**kwargs)
builtins.__import__=block
from vercor.setups import VerisConfig, replace_veris_parameters
assert callable(replace_veris_parameters)
assert not {'veris','veros'} & sys.modules.keys()
print(json.dumps({'lazy':True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout)["lazy"] is True


def density_coupler(density: float, *, authored: float | None = None) -> Any:
    from vercor import Coupler
    from vercor.physics import PhysicalConstants

    physical = {"h0": 0.43}
    if authored is not None:
        physical["rhoFresh"] = authored
    base, _, _, _ = small_coupler(config=configured(physical_constants=physical))
    return Coupler(
        base.clock,
        components=base.components.values(),
        exchanges=base.exchanges,
        run_order=base.run_order,
        runtime=base.runtime,
        constants=PhysicalConstants(freshwater_density=density),
    )


@pytest.mark.parametrize("density", [997.0, 1030.0])
def test_configured_density_round_trips_si_atmospheric_mass(density: float) -> None:
    from tests.test_veris_component import cold_fields
    from vercor.physics import PhysicalConstants
    from vercor.setups._external.veris_forcing import prepare_veris_forcing

    c = density_coupler(density, authored=density)
    initial = c.initial_state()
    p = native_payload(initial).physical_constants
    assert float(p.h0) == 0.43 and float(p.rhoFresh) == density
    mask = np.asarray(c.components["CICE"].grid.binary_mask)
    forcing = prepare_veris_forcing(
        cold_fields(),
        ocean_mask=mask,
        constants=PhysicalConstants(freshwater_density=density),
        dtype=c.runtime.dtype,
    )
    expected = np.asarray(cold_fields()["snow_rate"]).T * mask.T
    np.testing.assert_allclose(
        p.rhoFresh * forcing["snowfall"], expected, rtol=1e-14, atol=0
    )


def test_authored_density_conflict_is_rejected() -> None:
    with pytest.raises((ValueError, CouplerError), match="rhoFresh|freshwater|owned"):
        density_coupler(997.0, authored=1000.0).initial_state()


def test_step_density_follows_context_with_existing_payload() -> None:
    from vercor import Coupler
    from vercor.physics import PhysicalConstants

    first = density_coupler(997.0)
    initial = first.initial_state()
    second = Coupler(
        first.clock,
        components=first.components.values(),
        exchanges=first.exchanges,
        run_order=first.run_order,
        runtime=first.runtime,
        constants=PhysicalConstants(freshwater_density=1030.0),
    )
    expected = native_payload(first.run(initial))
    actual = native_payload(second.run(initial))
    assert float(actual.physical_constants.rhoFresh) == 1030.0
    assert float(native_payload(initial).physical_constants.rhoFresh) == 997.0
    for name in ("hIceMean", "hSnowMean", "Area", "TSurf"):
        np.testing.assert_allclose(
            getattr(actual.state, name),
            getattr(expected.state, name),
            rtol=1e-11,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        actual.diagnostics.ocean_mass_flux,
        expected.diagnostics.ocean_mass_flux,
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.fast_always
def test_new_physical_mapping_preserves_existing_config_positionals() -> None:
    from vercor.output import OutputSpec

    initial = {"Area": 0.5}
    output = OutputSpec()
    config = VerisConfig(
        "CICE", timedelta(seconds=300), {"nEVPsteps": 4}, initial, output
    )
    assert config.initial_conditions == initial
    assert config.output is output
    assert config.physical_constants is None
