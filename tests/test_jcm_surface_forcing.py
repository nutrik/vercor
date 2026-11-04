"""Exercise immutable forcing extension and the actual native SPEEDY hook."""

from dataclasses import FrozenInstanceError
from importlib import import_module
from inspect import signature
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor.dtypes import DTypePolicy
from vercor.setups._external._jax_gcm_pytree import tree_as_runtime_dtype

pytestmark = pytest.mark.fast_always


def hooks() -> Any:
    return import_module("vercor.setups._external.jax_gcm_surface")


def test_copy_and_time_selection_preserve_extension_and_native_fields() -> None:
    module = hooks()
    from jcm.forcing import ForcingData, make_time_series
    from jcm.date import DateData
    import jax_datetime as jdt

    shape = (3, 2)
    seconds = jdt.to_datetime("1970-01-01") + jdt.Timedelta(
        days=jnp.array([10951, 10952, 10953])
    )
    temperatures = jnp.stack(
        [jnp.full(shape, value) for value in (280.0, 290.0, 300.0)]
    )
    base = ForcingData.zeros(shape).copy(
        stl_am=make_time_series(temperatures, seconds),
        co2_vmr=make_time_series(jnp.array([360.0, 400.0, 450.0]), seconds),
        nudging_target={"marker": jnp.array(1.0)},
    )
    alpha = jnp.full(shape, 0.34)
    wrapped = module.CoupledJCMForcing(base, alpha)
    date = DateData.set_date(
        jdt.to_datetime("1970-01-01") + jdt.Timedelta(days=10952), dt_seconds=1800.0
    )
    actual = wrapped.select(date)
    expected = base.select(date)
    assert isinstance(actual, module.CoupledJCMForcing)
    np.testing.assert_allclose(actual.stl_am, expected.stl_am)
    np.testing.assert_allclose(actual.co2_vmr, expected.co2_vmr)
    np.testing.assert_allclose(actual.solar.tyear, expected.solar.tyear)
    np.testing.assert_allclose(actual.marine_surface_albedo, alpha)
    changed = wrapped.copy(
        stl_am=jnp.full(shape, 305.0),
        marine_surface_albedo=jnp.full(shape, 0.4),
        nudging_target=None,
    )
    np.testing.assert_allclose(changed.stl_am, 305.0)
    np.testing.assert_allclose(changed.marine_surface_albedo, 0.4)
    assert changed.nudging_target is None
    np.testing.assert_array_equal(base.stl_am.values, temperatures)
    np.testing.assert_allclose(wrapped.marine_surface_albedo, 0.34)
    np.testing.assert_allclose(wrapped.copy().select(date).marine_surface_albedo, 0.34)
    with pytest.raises(FrozenInstanceError):
        wrapped.marine_surface_albedo = alpha + 1


@pytest.mark.parametrize("enable_x64", [False, True])
def test_forcing_extension_is_a_normalizable_pytree(enable_x64: bool) -> None:
    module = hooks()
    from jcm.forcing import ForcingData

    value = module.CoupledJCMForcing(ForcingData.zeros((3, 2)), jnp.full((3, 2), 0.34))
    policy = DTypePolicy(enable_x64=enable_x64)
    normalized = tree_as_runtime_dtype(value, policy)
    assert isinstance(normalized, module.CoupledJCMForcing)
    leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(normalized)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
    ]
    assert leaves and all(jnp.asarray(leaf).dtype == policy.jax_real for leaf in leaves)
    np.testing.assert_allclose(normalized.marine_surface_albedo, 0.34, rtol=1e-6)


def native_case(*, resolution: int = 31) -> tuple[Any, Any, Any, Any]:
    """Construct actual native coordinate/state/forcing resources without files."""
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics_interface import PhysicsState
    from jcm.terrain import TerrainData
    from jcm.forcing import ForcingData

    coords = get_speedy_coords(spectral_truncation=resolution)
    shape = coords.horizontal.nodal_shape
    terrain = TerrainData.from_coords(
        coords, fmask=jnp.full(shape, 0.25), orography=jnp.zeros(shape), lfluxland=True
    )
    state = PhysicsState.zeros(
        (8, *shape),
        temperature=jnp.full((8, *shape), 280.0),
        normalized_surface_pressure=jnp.ones(shape),
    )
    forcing = ForcingData.zeros(shape).copy(
        alb0=jnp.full(shape, 0.2),
        snowc_am=jnp.full(shape, 0.25),
        sice_am=jnp.full(shape, 0.4),
        co2_vmr=jnp.array(450.0),
    )
    return coords, terrain, state, forcing


def test_plain_native_forcing_and_constructor_are_unchanged() -> None:
    module = hooks()
    from jcm.physics.speedy.speedy_terms import SpeedyForcing

    coords, terrain, state, forcing = native_case()
    old, new = SpeedyForcing(), module.CoupledSpeedyForcing()
    assert signature(type(new)) == signature(SpeedyForcing)
    old.cache_coords(coords)
    new.cache_coords(coords)
    expected = old(state, {}, forcing, terrain)
    actual = new(state, {}, forcing, terrain)
    assert cast(
        Any, jax.tree_util.tree_structure(actual)
    ) == jax.tree_util.tree_structure(expected)
    for left, right in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_array_equal(left, right)


def test_native_hook_uses_precomposed_marine_alpha_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = hooks()
    from jcm.physics.speedy.speedy_terms import SpeedyForcing

    coords, terrain, state, base = native_case()
    old = SpeedyForcing()
    old.cache_coords(coords)
    _, reference = old(state, {}, base, terrain)
    calls: list[bool] = []
    original = SpeedyForcing.__call__

    def recording(self: Any, *args: Any) -> Any:
        calls.append(True)
        return original(self, *args)

    monkeypatch.setattr(SpeedyForcing, "__call__", recording)
    term = module.CoupledSpeedyForcing()
    term.cache_coords(coords)
    shape = coords.horizontal.nodal_shape
    forcing = module.CoupledJCMForcing(base, jnp.full(shape, 0.34))
    _, result = term(state, {}, forcing, terrain)
    assert calls == [True]
    radcon = result["_mod_radcon"]
    assert cast(
        Any, jax.tree_util.tree_structure(result)
    ) == jax.tree_util.tree_structure(reference)
    assert isinstance(result["_forcing_2d"], type(base))
    np.testing.assert_allclose(radcon.alb_s, 0.34)
    np.testing.assert_allclose(
        radcon.albsfc, 0.34 + 0.25 * (reference["_mod_radcon"].alb_l - 0.34)
    )
    np.testing.assert_allclose(radcon.alb_l, reference["_mod_radcon"].alb_l)
    np.testing.assert_allclose(radcon.ablco2, reference["_mod_radcon"].ablco2)
    np.testing.assert_allclose(result["_forcing_2d"].sice_am, 0.4)
    assert not hasattr(term, "marine_surface_albedo")

    def loss(alpha: jax.Array) -> jax.Array:
        _, diagnostics = term(
            state,
            {},
            forcing.copy(marine_surface_albedo=jnp.full(shape, alpha)),
            terrain,
        )
        return cast(jax.Array, diagnostics["_mod_radcon"].albsfc.mean())

    point = jnp.array(0.34)
    _, forward = jax.jvp(jax.jit(loss), (point,), (jnp.array(1.0),))
    reverse = jax.grad(loss)(point)
    np.testing.assert_allclose([forward, reverse], 0.75, rtol=1e-12)
    np.testing.assert_allclose(
        reverse, (loss(point + 1e-4) - loss(point - 1e-4)) / 2e-4, rtol=1e-9
    )


def test_plain_native_model_bootstrap_retains_carry() -> None:
    module = hooks()
    from jcm.model import Model
    from jcm.physics.speedy.speedy_terms import speedy_physics

    coords, terrain, _, _ = native_case()
    ordinary = Model(
        coords=coords, terrain=terrain, physics=speedy_physics(), time_step=30.0
    )
    coupled = Model(
        coords=coords,
        terrain=terrain,
        physics=speedy_physics().replace("forcing", module.CoupledSpeedyForcing()),
        time_step=30.0,
    )
    ordinary.bootstrap_state()
    coupled.bootstrap_state()
    for attribute in ("_final_dycore_state", "_final_physics_state"):
        left, right = getattr(ordinary, attribute), getattr(coupled, attribute)
        assert cast(
            Any, jax.tree_util.tree_structure(left)
        ) == jax.tree_util.tree_structure(right)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
        ):
            np.testing.assert_array_equal(actual, expected)
