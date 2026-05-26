from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from vercor.types import RuntimeArray


def runtime_array_to_host(array: RuntimeArray) -> NDArray[Any]:
    """Transfer a runtime array to host memory for NumPy-only consumers."""

    return np.asarray(jax.device_get(jnp.asarray(array)))


def transposed_host_array(array: RuntimeArray) -> NDArray[Any]:
    """Transfer a transposed runtime array to host memory."""

    return runtime_array_to_host(jnp.asarray(array).T)
