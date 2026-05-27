from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import pytest

import vercor.host_arrays as host_arrays_module
from vercor.diagnostics import component_vector_speed
from vercor.host_arrays import transposed_host_array
from tests.assertions import assert_allclose_compact
from vercor.host_arrays import runtime_array_to_host
from vercor.runtime import RuntimeComponentState, RuntimeFieldStore


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
    state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(
            {
                "u": jnp.asarray([[3.0, 0.0], [4.0, 0.0]]),
                "v": jnp.asarray([[4.0, 0.0], [3.0, 0.0]]),
            }
        ),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
    )
    speed = component_vector_speed(state, "u", "v")

    assert isinstance(speed, jax.Array)
    assert_allclose_compact(speed, np.asarray([[5.0, 0.0], [5.0, 0.0]]))
