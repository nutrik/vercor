from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

import pytest

from tests.conftest import SelectFastCases
from vercor.clock import DateTime360
from vercor.setups.external.jax_gcm import _JAXGCMState


@dataclass(frozen=True)
class OutputFrequencyCase:
    case_id: str
    output_frequency: object | None
    time: datetime | DateTime360
    dt: timedelta
    expected: bool


def _make_component(output_frequency: object | None) -> Any:
    component = _JAXGCMState.__new__(_JAXGCMState)
    component.output_frequency = cast(Any, output_frequency)
    return component


@pytest.mark.fast_always
def test_should_write_output_frequency_cases(
    select_fast_cases: SelectFastCases,
) -> None:
    cases = [
        OutputFrequencyCase(
            case_id="none-always",
            output_frequency=None,
            time=datetime(2026, 2, 20, 0, 0, 0),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="day-boundary",
            output_frequency="day",
            time=datetime(2026, 2, 20, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="day-not-boundary",
            output_frequency="day",
            time=datetime(2026, 2, 20, 22, 0, 0),
            dt=timedelta(hours=1),
            expected=False,
        ),
        OutputFrequencyCase(
            case_id="day-boundary-case-insensitive",
            output_frequency="DAY",
            time=datetime(2026, 2, 20, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="month-boundary",
            output_frequency="month",
            time=datetime(2026, 2, 28, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="month-not-boundary",
            output_frequency="month",
            time=datetime(2026, 2, 27, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=False,
        ),
        OutputFrequencyCase(
            case_id="year-boundary",
            output_frequency="year",
            time=datetime(2026, 12, 31, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="year-not-boundary",
            output_frequency="year",
            time=datetime(2026, 12, 30, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=False,
        ),
        OutputFrequencyCase(
            case_id="invalid-frequency",
            output_frequency="hour",
            time=datetime(2026, 2, 20, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=False,
        ),
        OutputFrequencyCase(
            case_id="non-string-frequency",
            output_frequency=12,
            time=datetime(2026, 2, 20, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=False,
        ),
        OutputFrequencyCase(
            case_id="360-day-boundary",
            output_frequency="day",
            time=DateTime360(2026, 2, 20, 23, 0, 0, 0, 50),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="360-month-boundary",
            output_frequency="month",
            time=DateTime360(2026, 2, 30, 23, 0, 0, 0, 60),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="360-year-boundary",
            output_frequency="year",
            time=DateTime360(2026, 12, 30, 23, 0, 0, 0, 360),
            dt=timedelta(hours=1),
            expected=True,
        ),
    ]

    for case in select_fast_cases(
        cases,
        case_id=lambda case: case.case_id,
        min_cases=3,
    ):
        component = _make_component(case.output_frequency)
        assert component._should_write_output(case.time, case.dt) is case.expected


def test_is_period_end_stays_false_within_same_day() -> None:
    component = _make_component("day")
    time = datetime(2026, 2, 20, 12, 0, 0)

    assert component._is_period_end(time, timedelta(minutes=30), "day") is False
