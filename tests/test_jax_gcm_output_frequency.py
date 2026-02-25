from datetime import datetime, timedelta

from vercor.clock import DateTime360
from vercor.components.external.jax_gcm import JAXGCM


def _make_component(output_frequency: str | None) -> JAXGCM:
    component = JAXGCM.__new__(JAXGCM)
    component.output_frequency = output_frequency
    return component


def test_should_write_output_always_when_frequency_is_none() -> None:
    component = _make_component(None)

    assert component._should_write_output(
        datetime(2026, 2, 20, 0, 0, 0), timedelta(hours=1)
    )


def test_should_write_output_for_day_boundary_datetime() -> None:
    component = _make_component("day")

    assert component._should_write_output(
        datetime(2026, 2, 20, 23, 0, 0), timedelta(hours=1)
    )
    assert not component._should_write_output(
        datetime(2026, 2, 20, 22, 0, 0), timedelta(hours=1)
    )


def test_should_write_output_for_month_and_year_boundary_datetime() -> None:
    component = _make_component("month")

    assert component._should_write_output(
        datetime(2026, 2, 28, 23, 0, 0), timedelta(hours=1)
    )
    assert not component._should_write_output(
        datetime(2026, 2, 27, 23, 0, 0), timedelta(hours=1)
    )

    component.output_frequency = "year"
    assert component._should_write_output(
        datetime(2026, 12, 31, 23, 0, 0), timedelta(hours=1)
    )
    assert not component._should_write_output(
        datetime(2026, 12, 30, 23, 0, 0), timedelta(hours=1)
    )


def test_should_write_output_for_custom_datetime_boundaries() -> None:
    component = _make_component("day")
    model_time = DateTime360(
        year=2026,
        month=2,
        day=20,
        hour=23,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=50,
    )
    assert component._should_write_output(model_time, timedelta(hours=1))

    component.output_frequency = "month"
    month_end_time = DateTime360(
        year=2026,
        month=2,
        day=30,
        hour=23,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=60,
    )
    assert component._should_write_output(month_end_time, timedelta(hours=1))

    component.output_frequency = "year"
    year_end_time = DateTime360(
        year=2026,
        month=12,
        day=30,
        hour=23,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=360,
    )
    assert component._should_write_output(year_end_time, timedelta(hours=1))


def test_should_not_write_output_for_invalid_frequency() -> None:
    component = _make_component("hour")

    assert not component._should_write_output(
        datetime(2026, 2, 20, 23, 0, 0), timedelta(hours=1)
    )
