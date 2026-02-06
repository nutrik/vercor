from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from vercor.clock import Clock
from vercor.settings import VercorSettings
from vercor.tools import (
    get_field_at_specific_time,
    get_periodic_interval,
    get_field_time_slice,
)


@dataclass
class DummyCoupler:
    clock: Clock
    settings: VercorSettings


def make_coupler(year_in_seconds: float) -> DummyCoupler:
    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=1.0, steps=1)
    settings = VercorSettings(year_in_seconds=year_in_seconds)
    return DummyCoupler(clock=clock, settings=settings)


def test_get_field_at_specific_time_weights_and_interpolation() -> None:
    coupler = make_coupler(year_in_seconds=12.0)

    # Build a (lat, lon, time) array with constant values per record
    lat, lon, nrec = 2, 3, 12
    arr = np.zeros((lat, lon, nrec), dtype=float)
    arr[..., 0] = 0.0
    arr[..., 1] = 10.0
    data = {"foo": arr}

    current_time = coupler.clock.start + timedelta(seconds=0.25)

    total_seconds = (current_time - coupler.clock.start).total_seconds()
    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=total_seconds,
        cycle_length=coupler.settings.year_in_seconds,
        rec_spacing=coupler.settings.year_in_seconds / 12.0,
        n_rec=12,
    )

    assert n1 == 0
    assert n2 == 1
    assert np.isclose(f1 + f2, 1.0)

    out = get_field_at_specific_time("foo", data, coupler, current_time=current_time)  # type: ignore
    assert np.allclose(out, 2.5)


def test_get_field_at_specific_time_boundary_record() -> None:
    coupler = make_coupler(year_in_seconds=120.0)

    lat, lon, nrec = 2, 2, 12
    arr = np.zeros((lat, lon, nrec), dtype=float)
    arr[..., 1] = 7.0
    data = {"foo": arr}

    rec_spacing = coupler.settings.year_in_seconds / 12.0
    current_time = coupler.clock.start + timedelta(seconds=rec_spacing)

    out = get_field_at_specific_time("foo", data, coupler, current_time=current_time)  # type: ignore
    assert np.allclose(out, 7.0)


def test_get_field_at_specific_time_axis_ordering() -> None:
    coupler = make_coupler(year_in_seconds=12.0)

    # lat-major data; output is expected to be (lon, lat) due to swapaxes
    arr = np.zeros((2, 3, 12), dtype=float)
    arr[:, :, 0] = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    data = {"foo": arr}

    out = get_field_at_specific_time("foo", data, coupler, current_time=coupler.clock.start)  # type: ignore
    expected = arr[:, :, 0].swapaxes(-2, -1)

    assert out.shape == (3, 2)
    assert np.allclose(out, expected)


def test_get_field_time_slice_basic_indexing() -> None:
    data = {"foo": np.arange(365 * 2, dtype=float).reshape(365, 2)}

    time = datetime(2001, 1, 1)
    out = get_field_time_slice("foo", data, time)
    assert np.allclose(out, data["foo"][0, :])

    time = datetime(2001, 12, 31)
    out = get_field_time_slice("foo", data, time)
    assert np.allclose(out, data["foo"][364, :])


def test_get_field_time_slice_no_leap_year_adjustment() -> None:
    data = {"foo": np.arange(366, dtype=float)}

    time = datetime(2000, 2, 29)
    out = get_field_time_slice("foo", data, time, no_leap=True)

    # Feb 29 in a leap year is mapped to Feb 28 when no_leap=True
    assert np.isclose(out, data["foo"][58])


def test_get_field_time_slice_leap_day_retained_when_requested() -> None:
    data = {"foo": np.arange(366, dtype=float)}

    time = datetime(2000, 2, 29)
    out = get_field_time_slice("foo", data, time, no_leap=False)

    assert np.isclose(out, data["foo"][59])
