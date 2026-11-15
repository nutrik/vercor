"""Bound driver identity checks without weakening custom-backend rejection."""

from __future__ import annotations

from typing import Any, cast

import pytest

from vercor._runtime.backends import _RuntimeDriverAdapter
from vercor.exceptions import CouplerError
from vercor.runtime import ExecutionChunk, StepPlan

pytestmark = pytest.mark.fast_always


def test_valid_driver_plans_do_not_copy_consumed_prefixes() -> None:
    """A long valid chunk must not allocate quadratically growing prefixes."""
    slices: list[slice] = []

    class ObservedPlans(tuple[StepPlan, ...]):
        def __getitem__(self, key: Any) -> Any:
            if isinstance(key, slice):
                slices.append(key)
            return super().__getitem__(key)

    plans = ObservedPlans(StepPlan(index, ("A",)) for index in range(1024))
    chunk = ExecutionChunk(plans)
    # Instrument the real immutable tuple access; ordinary construction owns it.
    object.__setattr__(chunk, "steps", plans)
    driver = _RuntimeDriverAdapter(cast(Any, None), chunk, cast(Any, None))
    for plan in plans:
        assert driver._expected_plan(plan) is plan
        driver._cursor += 1
    assert len(slices) == 0, f"ERROR valid dispatch copied {len(slices)} plan prefixes"


@pytest.mark.parametrize(
    ("cursor", "selection", "message"),
    [
        (1, "equal-copy", "outside the active execution chunk"),
        (3, "last", "already executed"),
        (3, "foreign", "outside the active execution chunk"),
        (0, "invalid-type", "must be a StepPlan"),
    ],
)
def test_driver_identity_errors_survive_fast_path(
    cursor: int, selection: str, message: str
) -> None:
    """Equal values cannot forge identities, including after chunk exhaustion."""
    plans = tuple(StepPlan(index, ("A",)) for index in range(3))
    driver = _RuntimeDriverAdapter(
        cast(Any, None), ExecutionChunk(plans), cast(Any, None)
    )
    driver._cursor = cursor
    if selection == "last":
        selected: Any = plans[-1]
    elif selection == "equal-copy":
        selected = StepPlan(1, ("A",))
    elif selection == "invalid-type":
        selected = object()
    else:
        selected = StepPlan(3, ("A",))
    with pytest.raises(CouplerError, match=message):
        driver._expected_plan(selected)
