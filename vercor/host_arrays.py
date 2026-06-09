from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from vercor.types import RuntimeArray


def array_to_host(value: Any) -> NDArray[Any]:
    """Transfer an array-like value to host memory for I/O-only consumers."""

    return np.asarray(jax.device_get(value))


def host_int64_array(value: Any) -> NDArray[Any]:
    """Return a host ``int64`` array for file formats requiring wide integers.

    This intentionally stays outside JAX because ``jax_enable_x64=False`` can
    make large calendar offsets overflow before NetCDF writing.
    """

    return np.asarray(value, dtype=np.int64)


def runtime_array_to_host(array: RuntimeArray) -> NDArray[Any]:
    """Transfer a runtime array to host memory for NumPy-only consumers."""

    return array_to_host(jnp.asarray(array))


def transposed_host_array(array: RuntimeArray) -> NDArray[Any]:
    """Transfer a transposed runtime array to host memory."""

    return runtime_array_to_host(jnp.asarray(array).T)
