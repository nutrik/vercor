from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np


class SupportsEnableX64(Protocol):
    """Settings-like object exposing VerCOR's real-precision switch."""

    @property
    def enable_x64(self) -> bool:
        """Return whether 64-bit real arrays are enabled."""
        ...


ShapeLike: TypeAlias = int | Sequence[int]


@dataclass(frozen=True)
class DTypePolicy:
    """Canonical dtype choices for VerCOR-owned arrays.

    The real dtype follows the configured ``enable_x64`` precision switch. Index
    arrays intentionally stay 32-bit to keep sparse indexing and runtime
    metadata compact across both real-precision modes.
    """

    enable_x64: bool

    @classmethod
    def from_jax_config(cls) -> "DTypePolicy":
        """Return a policy matching the active JAX global precision setting."""

        return cls(enable_x64=bool(jax.config.read("jax_enable_x64")))

    @classmethod
    def from_settings(cls, settings: SupportsEnableX64) -> "DTypePolicy":
        """Return a policy from a settings-like object."""

        return cls(enable_x64=bool(settings.enable_x64))

    @property
    def jax_real(self) -> Any:
        """Return the canonical JAX real dtype."""

        return jnp.float64 if self.enable_x64 else jnp.float32

    @property
    def jax_index(self) -> Any:
        """Return the canonical JAX index dtype."""

        return jnp.int32

    @property
    def numpy_real(self) -> np.dtype[Any]:
        """Return the canonical NumPy real dtype."""

        return np.dtype(np.float64 if self.enable_x64 else np.float32)

    @property
    def numpy_index(self) -> np.dtype[Any]:
        """Return the canonical NumPy index dtype."""

        return np.dtype(np.int32)


PrecisionPolicy: TypeAlias = DTypePolicy | SupportsEnableX64 | None


def dtype_policy(policy: PrecisionPolicy = None) -> DTypePolicy:
    """Normalize optional settings/policy input into a ``DTypePolicy``."""

    if policy is None:
        return DTypePolicy.from_jax_config()
    if isinstance(policy, DTypePolicy):
        return policy
    return DTypePolicy.from_settings(policy)


def jax_real_dtype(policy: PrecisionPolicy = None) -> Any:
    """Return the canonical JAX real dtype for ``policy``."""

    return dtype_policy(policy).jax_real


def jax_index_dtype(policy: PrecisionPolicy = None) -> Any:
    """Return the canonical JAX index dtype for ``policy``."""

    return dtype_policy(policy).jax_index


def as_jax_real_array(value: Any, policy: PrecisionPolicy = None) -> jax.Array:
    """Convert ``value`` to a JAX array using VerCOR's real dtype policy."""

    if policy is None:
        return jnp.asarray(value)
    return jnp.asarray(value, dtype=jax_real_dtype(policy))


def as_jax_index_array(value: Any, policy: PrecisionPolicy = None) -> jax.Array:
    """Convert ``value`` to a JAX array using VerCOR's canonical index dtype."""

    return jnp.asarray(value, dtype=jax_index_dtype(policy))


def jax_zeros(shape: ShapeLike, policy: PrecisionPolicy = None) -> jax.Array:
    """Return zeros with VerCOR's canonical real dtype."""

    return jnp.zeros(shape, dtype=jax_real_dtype(policy))


def jax_ones(shape: ShapeLike, policy: PrecisionPolicy = None) -> jax.Array:
    """Return ones with VerCOR's canonical real dtype."""

    return jnp.ones(shape, dtype=jax_real_dtype(policy))


def jax_full(
    shape: ShapeLike,
    fill_value: Any,
    policy: PrecisionPolicy = None,
) -> jax.Array:
    """Return a full array with VerCOR's canonical real dtype."""

    return jnp.full(shape, fill_value, dtype=jax_real_dtype(policy))


def jax_linspace(
    start: Any,
    stop: Any,
    num: int,
    policy: PrecisionPolicy = None,
) -> jax.Array:
    """Return evenly spaced values with VerCOR's canonical real dtype."""

    return jnp.linspace(start, stop, num, dtype=jax_real_dtype(policy))


def jax_arange(
    start: Any,
    stop: Any | None = None,
    step: Any = 1,
    policy: PrecisionPolicy = None,
) -> jax.Array:
    """Return evenly spaced values with VerCOR's canonical real dtype."""

    if stop is None:
        return jnp.arange(start, dtype=jax_real_dtype(policy))
    return jnp.arange(start, stop, step, dtype=jax_real_dtype(policy))
