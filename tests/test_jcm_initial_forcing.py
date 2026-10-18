"""Evaluate real initial atmosphere forcing without advancing native carries."""

from datetime import datetime, timedelta
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor.dtypes import DTypePolicy
from vercor.physics import PhysicalConstants
from vercor.setups import Spinup
from tests.test_jcm_surface_runtime import make_native_jcm

pytestmark = pytest.mark.fast_always


def assert_same_tree(left: Any, right: Any) -> None:
    assert cast(
        Any, jax.tree_util.tree_structure(left)
    ) == jax.tree_util.tree_structure(right)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "start,expected",
    [
        (datetime(2000, 1, 1), 10950),
        (datetime(2000, 1, 3), 10952),
        (datetime(2000, 3, 1), 11009),
        (datetime(2001, 1, 1), 11315),
    ],
)
def test_native_noleap_epoch_matches_coupler(start: datetime, expected: int) -> None:
    from vercor.setups._external.jax_gcm_state import jax_gcm_noleap_epoch
    from jcm.date import DateData

    epoch = jax_gcm_noleap_epoch(start, timedelta(0))
    assert int(epoch.delta.days) == expected
    date = DateData.set_date(epoch)
    model_year = date.model_year("365_day")
    assert int(model_year) == start.year
    if start.month == 1 and start.day == 3:
        np.testing.assert_allclose(date.tyear("365_day"), 2 / 365)


def test_initial_forcing_does_not_advance_native_carry() -> None:
    from vercor.setups._external.jax_gcm_runtime import diagnose_jax_gcm_initial_fields
    from vercor.setups._external._jax_gcm_pytree import tree_as_runtime_dtype

    coupler, _, owner = make_native_jcm()
    initial = coupler.initial_state()
    payload = cast(Any, initial._component_state("ATM").payload)
    original_dycore = payload.jcm_state.dycore_state
    original_carry = payload.jcm_state.physics_carry
    fields, evaluated = diagnose_jax_gcm_initial_fields(
        owner, payload, PhysicalConstants(), DTypePolicy(enable_x64=True)
    )
    assert_same_tree(evaluated.jcm_state.dycore_state, original_dycore)
    assert_same_tree(evaluated.jcm_state.physics_carry, original_carry)
    np.testing.assert_array_equal(owner.model.dycore.sim_time(original_dycore), 0)
    assert_same_tree(
        payload.jcm_state.dycore_state,
        tree_as_runtime_dtype(
            owner.model._final_dycore_state, DTypePolicy(enable_x64=True)
        ),
    )
    assert_same_tree(
        payload.jcm_state.physics_carry,
        tree_as_runtime_dtype(
            owner.model._final_physics_state, DTypePolicy(enable_x64=True)
        ),
    )
    current = owner.model.dycore.to_physics_state(original_dycore)
    date = owner.model._date_from_sim_time(owner.model.dycore.sim_time(original_dycore))
    selected = payload.forcing.select(date, calendar=owner.model.calendar)
    _, direct = owner.model.physics.compute_tendencies(
        current, selected, owner.model.terrain, prev_physics_data=original_carry
    )
    expected = direct["_shortwave_rad"].rsds.T
    assert bool(jnp.any(expected > 0))
    np.testing.assert_allclose(fields["downward_shortwave_radiation_flux"], expected)
    np.testing.assert_allclose(
        initial.component("ATM").field("downward_shortwave_radiation_flux"), expected
    )
    np.testing.assert_allclose(fields["open_water_albedo"], 0.1)
    assert_same_tree(
        evaluated.jcm_state.physics,
        tree_as_runtime_dtype(direct, DTypePolicy(enable_x64=True)),
    )


def test_two_day_spinup_ends_at_the_native_coupling_epoch() -> None:
    coupler, _, owner = make_native_jcm(
        start=datetime(2000, 1, 3),
        coupling_seconds=86400,
        spinup=Spinup(enabled=True, duration=timedelta(days=2)),
    )
    initial = coupler.initial_state()
    payload = cast(Any, initial._component_state("ATM").payload)
    assert int(owner.model.start_date.delta.days) == 10950
    date = owner.model._date_from_sim_time(
        owner.model.dycore.sim_time(payload.jcm_state.dycore_state)
    )
    assert int(date.model_year("365_day")) == 2000
    np.testing.assert_allclose(date.tyear("365_day"), 2 / 365)


def test_disabled_spinup_ignores_its_duration_and_conflicting_epoch_is_rejected() -> (
    None
):
    coupler, component, owner = make_native_jcm(
        start=datetime(2000, 1, 3),
        coupling_seconds=86400,
        spinup=Spinup(enabled=False, duration=timedelta(hours=36)),
    )
    coupler.initial_state()
    assert int(owner.model.start_date.delta.days) == 10952
    from vercor.components import SetupContext

    context = SetupContext(
        datetime(2001, 1, 3),
        86400,
        ("ATM",),
        coupler.logger,
        dtype=DTypePolicy(enable_x64=True),
    )
    with pytest.raises(Exception, match="epoch|start|new.*component"):
        owner.setup(component, context)


def test_enabled_spinup_must_align_to_coupling_intervals() -> None:
    coupler, _, _ = make_native_jcm(
        coupling_seconds=86400,
        spinup=Spinup(enabled=True, duration=timedelta(hours=36)),
    )
    with pytest.raises(ValueError, match="spinup|coupling"):
        coupler.initial_state()


def test_repeated_runs_use_the_supplied_payload_independently() -> None:
    coupler, _, owner = make_native_jcm()
    initial = coupler.initial_state()
    original = cast(Any, initial._component_state("ATM").payload)
    first = coupler.run(initial)
    second = coupler.run(initial)
    assert_same_tree(
        first._component_state("ATM").payload, second._component_state("ATM").payload
    )
    assert_same_tree(initial._component_state("ATM").payload, original)
    assert float(owner.model.dycore.sim_time(original.jcm_state.dycore_state)) == 0
