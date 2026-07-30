from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import pytest

from vercor.setups.gallery import custom_component_wrapping
import vercor._host_arrays as host_arrays_module
from vercor.diagnostics import component_vector_speed
from vercor.dtypes import DTypePolicy
from vercor.grids import RectilinearGrid
from vercor._host_arrays import transposed_host_array
from tests.assertions import assert_allclose_compact
from vercor._host_arrays import runtime_array_to_host
from vercor.state import ComponentState


def test_custom_component_example_runs_behaviorally() -> None:
    grid = custom_component_wrapping.make_example_grid()
    coupler = custom_component_wrapping.make_custom_coupler(
        grid,
        log_level="warning",
        dtype=DTypePolicy(enable_x64=False),
    )

    final_state = coupler.run()

    assert_allclose_compact(
        final_state.component("MODEL").field("custom_flux"),
        jnp.full(grid.shape, 3.0),
    )
    assert coupler.log_level == "warning"
    assert coupler.runtime.dtype == DTypePolicy(enable_x64=False)
    assert callable(custom_component_wrapping.run_setup)


@pytest.mark.fast_always
def test_custom_component_run_setup_maps_cli_options_without_running_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(custom_component_wrapping, "make_example_grid", lambda: grid)
    monkeypatch.setattr(
        custom_component_wrapping,
        "make_data_forcing",
        lambda actual_grid: ("forcing", actual_grid),
    )
    monkeypatch.setattr(
        custom_component_wrapping,
        "make_differentiable_model",
        lambda actual_grid: ("model", actual_grid),
    )
    monkeypatch.setattr(
        custom_component_wrapping,
        "make_host_model",
        lambda actual_grid: ("host", actual_grid),
    )

    def fake_make_custom_coupler(
        actual_grid: object,
        *,
        log_level: int | str,
        dtype: DTypePolicy,
    ) -> str:
        captured.update(
            grid=actual_grid,
            log_level=log_level,
            dtype=dtype,
        )
        return "coupler"

    monkeypatch.setattr(
        custom_component_wrapping,
        "make_custom_coupler",
        fake_make_custom_coupler,
    )

    result = custom_component_wrapping.run_setup(
        loglevel="error",
        float_type="float32",
    )

    assert result is None
    assert captured == {
        "grid": grid,
        "log_level": "error",
        "dtype": DTypePolicy(enable_x64=False),
    }


def test_runtime_array_to_host_is_canonical_host_transfer() -> None:
    host_array = runtime_array_to_host(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))

    assert isinstance(host_array, np.ndarray)
    assert_allclose_compact(host_array, np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_transposed_host_array_uses_canonical_host_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[NDArray[Any]] = []

    def fake_runtime_array_to_host(array: object) -> NDArray[Any]:
        calls.append(np.asarray(array))
        return np.asarray(array)

    monkeypatch.setattr(
        host_arrays_module, "runtime_array_to_host", fake_runtime_array_to_host
    )
    host_array = transposed_host_array(jnp.asarray([[1, 2, 3], [4, 5, 6]]))

    assert isinstance(host_array, np.ndarray)
    assert len(calls) == 1
    assert_allclose_compact(calls[0], np.asarray([[1, 4], [2, 5], [3, 6]]))
    assert_allclose_compact(host_array, np.asarray([[1, 4], [2, 5], [3, 6]]))


def test_component_vector_speed_uses_jax_arrays() -> None:
    state = ComponentState(
        name="ATM",
        grid=None,
        fields={
            "u": jnp.asarray([[3.0, 0.0], [4.0, 0.0]]),
            "v": jnp.asarray([[4.0, 0.0], [3.0, 0.0]]),
        },
    )
    speed = component_vector_speed(state, "u", "v")

    assert isinstance(speed, jax.Array)
    assert_allclose_compact(speed, np.asarray([[5.0, 0.0], [5.0, 0.0]]))


def test_component_vector_speed_reads_runtime_component_view() -> None:
    grid = RectilinearGrid(
        "dummy",
        longitude=np.array([0.0, 1.0]),
        latitude=np.array([0.0, 1.0]),
    )
    view = ComponentState(
        name="ATM",
        grid=grid,
        received={
            "u": jnp.asarray([[5.0, 0.0], [12.0, 0.0]]),
            "v": jnp.asarray([[12.0, 0.0], [5.0, 0.0]]),
        },
    )

    speed = component_vector_speed(view, "u", "v")

    assert isinstance(speed, jax.Array)
    assert_allclose_compact(speed, np.asarray([[13.0, 0.0], [13.0, 0.0]]))
