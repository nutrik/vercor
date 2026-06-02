from __future__ import annotations

import logging
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vercor.grid_masks as grid_masks_module

from tests._coverage_support import capture_logger_output
from tests.assertions import assert_allclose_compact, assert_array_equal_compact
from vercor.exceptions import RegridderError
from vercor.grid_masks import (
    check_remap_conservation,
    check_total_lnd_ocn_mask_sum,
    compute_ocn_lnd_masks_on_atm_grid,
    create_lnd_mask_from_ocn,
)
from vercor.grid import RectilinearGrid
from vercor.jax_logging import DEFAULT_LOGGER_NAME


def test_compute_ocn_lnd_masks_on_atm_grid_clips_and_builds_binary_land_mask() -> None:
    class DummyRegridder:
        def __call__(self, _arr: np.ndarray) -> jax.Array:
            return jnp.asarray([[1.2, -0.2], [0.4, 0.0]])

    ocean_binary_mask = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    ocn_fmask, lnd_fmask, lnd_bmask = compute_ocn_lnd_masks_on_atm_grid(
        ocean_binary_mask,
        cast(Any, DummyRegridder()),
    )

    assert_allclose_compact(ocn_fmask, np.array([[1.0, 0.0], [0.4, 0.0]]))
    assert_allclose_compact(lnd_fmask, np.array([[0.0, 1.0], [0.6, 1.0]]))
    assert_array_equal_compact(lnd_bmask, np.array([[0, 1], [1, 1]]))


def test_check_total_lnd_ocn_mask_sum_success_and_failure() -> None:
    lnd_good = jnp.asarray([[0.3, 1.0], [0.0, 0.8]])
    ocn_good = jnp.asarray([[0.7, 0.0], [1.0, 0.2]])
    check_total_lnd_ocn_mask_sum(lnd_good, ocn_good)

    lnd_bad = jnp.asarray([[0.3, 1.0], [0.0, 0.8]])
    ocn_bad = jnp.asarray([[0.7, 0.0], [1.0, 0.0]])
    with pytest.raises(RegridderError, match="must sum to approx. 1"):
        check_total_lnd_ocn_mask_sum(lnd_bad, ocn_bad)


@pytest.mark.fast_always
def test_check_remap_conservation_handles_skip_and_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyRemapper:
        def __init__(
            self,
            src_lat_b: np.ndarray,
            dst_lat_b: np.ndarray,
            src_mass: float,
            dst_mass: float,
        ) -> None:
            self.src_lat_b = src_lat_b
            self.dst_lat_b = dst_lat_b
            self._src_mass = src_mass
            self._dst_mass = dst_mass

        def get_src_total_mass(self, _arr: np.ndarray) -> float:
            return self._src_mass

        def get_dst_total_mass(self, _arr: np.ndarray) -> float:
            return self._dst_mass

    class DummyRegridder:
        def __init__(self, interpolator: Any) -> None:
            self.interpolator = interpolator

    monkeypatch.setattr(
        grid_masks_module, "ConservativeRectilinearRemapper", DummyRemapper
    )
    skip_interp = DummyRemapper(
        src_lat_b=np.array([-90.0, 0.0, 90.0]),
        dst_lat_b=np.array([-80.0, 0.0, 80.0]),
        src_mass=10.0,
        dst_mass=1.0,
    )
    with capture_logger_output(DEFAULT_LOGGER_NAME, level=logging.WARNING) as stream:
        check_remap_conservation(
            cast(Any, DummyRegridder(skip_interp)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        )
    assert "Skipping mass conservation check" in stream.getvalue()

    mismatch_interp = DummyRemapper(
        src_lat_b=np.array([-90.0, 0.0, 90.0]),
        dst_lat_b=np.array([-90.0, 0.0, 90.0]),
        src_mass=10.0,
        dst_mass=9.0,
    )
    with pytest.raises(RegridderError, match="does not conserve total mass"):
        check_remap_conservation(
            cast(Any, DummyRegridder(mismatch_interp)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        )


def test_create_lnd_mask_from_ocn_accepts_jax_backed_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyRegridder:
        def __init__(
            self, source_grid: RectilinearGrid, destination_grid: RectilinearGrid
        ):
            self.source_grid = source_grid
            self.destination_grid = destination_grid
            self.interpolator = None

        def __call__(self, _field: jax.Array) -> jax.Array:
            return jnp.asarray([[0.8, 0.1], [0.0, 0.6]])

    monkeypatch.setattr(
        grid_masks_module,
        "ConservativeRectilinearRegridder",
        DummyRegridder,
    )

    ocn_grid = RectilinearGrid(
        name="OCN",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
        binary_mask=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
    )

    lnd_bmask, lnd_fmask = create_lnd_mask_from_ocn(
        atm_lat=jnp.asarray([0.0, 1.0]),
        atm_lon=jnp.asarray([0.0, 1.0]),
        ocn_grid=ocn_grid,
    )

    assert_array_equal_compact(lnd_bmask, np.asarray([[1, 1], [1, 1]]))
    assert_allclose_compact(lnd_fmask, np.asarray([[0.2, 0.9], [1.0, 0.4]]))
