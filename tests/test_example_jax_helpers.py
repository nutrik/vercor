import jax
import jax.numpy as jnp
import numpy as np

from setups.jax_array_helpers import (
    component_vector_speed,
    to_host_array,
    transposed_host_array,
)
from tests.assertions import assert_allclose_compact
from vercor.runtime import RuntimeComponentState, RuntimeFieldStore


def test_to_host_array_transfers_jax_array_to_host_array() -> None:
    host_array = to_host_array(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))

    assert isinstance(host_array, np.ndarray)
    assert_allclose_compact(host_array, np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_transposed_host_array_transfers_transposed_runtime_array() -> None:
    host_array = transposed_host_array(jnp.asarray([[1, 2, 3], [4, 5, 6]]))

    assert isinstance(host_array, np.ndarray)
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
