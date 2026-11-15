"""Profiling modes must measure the selected runtime without hiding host work."""

from __future__ import annotations

from typing import Any

import pytest

from vercor.dtypes import DTypePolicy
from vercor.setups.gallery import profile_runtime as profiling

pytestmark = pytest.mark.fast_always


def test_profile_parser_selects_backend_and_invocation() -> None:
    """The CLI must expose host/API costs as well as the compiled control."""
    args = profiling.build_parser().parse_args(
        ["--backend", "host", "--invocation", "api", "--json"]
    )
    assert (args.backend, args.invocation, args.json) == ("host", "api", True)


@pytest.mark.parametrize(
    ("backend", "invocation", "resolved"),
    [("host", None, "api"), ("host", "api", "api"), ("jax", "api", "api")],
)
def test_profile_runs_real_selected_backend(
    backend: Any, invocation: Any, resolved: str
) -> None:
    """The measured path must accept host execution without an outer trace."""
    result = profiling.profile_runtime(
        steps=1,
        grid_nx=4,
        grid_ny=4,
        log_level="ERROR",
        dtype=DTypePolicy(enable_x64=True),
        repeats=2,
        backend=backend,
        invocation=invocation,
    )
    assert result.backend == backend
    assert result.invocation == resolved
    assert result.setup_seconds >= 0
    assert result.run_seconds >= 0
    assert len(result.warm_run_seconds) == 2
    assert all(value >= 0 for value in result.warm_run_seconds)
    assert result.final_state_leaves > 0


def test_profile_builder_forces_host_backend() -> None:
    """Selecting host must reach runtime policy rather than label a JAX run."""
    coupler = profiling.build_slab_coupler(
        steps=1, grid_nx=4, grid_ny=4, log_level="ERROR", backend="host"
    )
    assert coupler.runtime.backend == "host"


def test_profile_rejects_outer_jit_of_host() -> None:
    """Host profiling cannot silently stage the measured Python driver."""
    with pytest.raises(ValueError, match="outer-jit.*jax"):
        profiling.profile_runtime(
            steps=1,
            grid_nx=4,
            grid_ny=4,
            log_level="ERROR",
            backend="host",
            invocation="outer-jit",
        )


def test_profile_default_retains_outer_jit_control() -> None:
    """Existing callers retain the original compiled-runtime measurement."""
    result = profiling.profile_runtime(
        steps=1, grid_nx=4, grid_ny=4, log_level="ERROR", repeats=1
    )
    assert (result.backend, result.invocation) == ("jax", "outer-jit")
