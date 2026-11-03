"""Keep released Veros numerical backend independent of component execution."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from vercor.setups import VerosConfig
from vercor.setups._external import veros_runtime_settings

pytestmark = pytest.mark.fast_always


@pytest.mark.parametrize(
    ("execution", "backend", "solver"),
    (("host", "numpy", "best"), ("host", "jax", "best"), ("jax", "jax", "scipy_jax")),
)
def test_explicit_native_backend_selects_runtime_without_changing_execution(
    monkeypatch: pytest.MonkeyPatch, execution: Any, backend: Any, solver: str
) -> None:
    """Catch host execution silently forcing NumPy despite an explicit choice."""
    settings = SimpleNamespace()
    module = ModuleType("veros")
    module.runtime_settings = settings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "veros", module)

    config = VerosConfig(execution=execution, backend=backend)
    veros_runtime_settings.configure_veros_runtime(
        config.execution, backend=config.backend
    )

    assert settings.backend == backend
    assert settings.linear_solver == solver
    assert config.execution == execution


@pytest.mark.parametrize("backend", ("unknown", 3, True))
def test_native_backend_rejects_invalid_configuration(backend: Any) -> None:
    """Catch an invalid backend reaching native settings mutation."""
    with pytest.raises(ValueError, match="backend"):
        VerosConfig(backend=backend)


def test_scanned_execution_rejects_numpy_before_native_configuration() -> None:
    """A NumPy payload cannot be accepted as a scanned JAX component."""
    with pytest.raises(ValueError, match="JAX|jax"):
        VerosConfig(execution="jax", backend="numpy")


def test_explicit_backend_is_configured_before_implementation_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch import-time backend locking and a stock-host fork requirement."""
    from vercor.setups import make_veros_gcm
    from vercor.setups._external import veros_gcm

    events: list[object] = []

    class ImplementationReached(RuntimeError):
        pass

    def configure(execution: str, *, backend: str | None = None) -> None:
        events.append((execution, backend))

    def require_fork() -> None:
        pytest.fail("Stock JAX kernels must not require the differentiable fork")

    def load() -> Any:
        events.append("implementation")
        raise ImplementationReached

    monkeypatch.setattr(veros_runtime_settings, "configure_veros_runtime", configure)
    monkeypatch.setattr(
        veros_runtime_settings,
        "require_differentiable_veros_capabilities",
        require_fork,
    )
    monkeypatch.setattr(veros_gcm, "_load_veros_implementation", load)
    with pytest.raises(ImplementationReached):
        make_veros_gcm(config=VerosConfig(execution="host", backend="jax"))
    assert events == [("host", "jax"), "implementation"]


def test_native_backend_conflict_does_not_silently_switch_a_locked_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch silently accepting a previous process-wide NumPy initialization."""

    class Locked:
        backend = "numpy"

        def __setattr__(self, name: str, value: object) -> None:
            raise RuntimeError("locked")

    module = ModuleType("veros")
    module.runtime_settings = Locked()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "veros", module)
    with pytest.raises(RuntimeError, match="separate Python processes"):
        veros_runtime_settings.configure_veros_runtime("host", backend="jax")


def test_stock_jax_tke_forcing_accepts_calm_and_windy_stress(tmp_path: Path) -> None:
    """Exercise the real stock backend without importing fork-only safe_sqrt."""
    code = """
import jax
import jax.numpy as jnp
import numpy as np
from vercor.setups._external.veros_runtime_settings import configure_veros_runtime
configure_veros_runtime('host', backend='jax')
from vercor.setups._external.veros_setup import _surface_tke_forcing
actual = jax.jit(_surface_tke_forcing)(jnp.array([0., 3.]), jnp.array([0., 4.]), 1.)
np.testing.assert_allclose(actual, [0., 5. ** 1.5], rtol=1e-12, atol=0.)
assert np.all(np.isfinite(actual))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "JAX_ENABLE_X64": "true"},
        capture_output=True,
        text=True,
        check=False,
    )
    log = tmp_path / "stock-jax-tke.log"
    log.write_text(result.stdout + result.stderr)
    assert result.returncode == 0, f"{result.stderr[-2000:]}\nFull log: {log}"


def test_native_backend_validation_does_not_mutate_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject contradictory direct private calls before setting any native flag."""
    settings = SimpleNamespace()
    module = ModuleType("veros")
    module.runtime_settings = settings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "veros", module)
    with pytest.raises(ValueError, match="JAX|jax"):
        veros_runtime_settings.configure_veros_runtime("jax", backend="numpy")
    assert vars(settings) == {}
    with pytest.raises(ValueError, match="backend"):
        veros_runtime_settings.configure_veros_runtime("host", backend=cast(Any, "bad"))
    assert vars(settings) == {}


@pytest.mark.parametrize("selection", ("default", "numpy"))
def test_default_and_explicit_numpy_run_actual_native_backend(
    selection: str, tmp_path: Path
) -> None:
    """Prove automatic JAX selection and the explicit NumPy escape hatch work."""
    code = f"""
from datetime import datetime
import jax
import numpy as np
from vercor import Clock, Coupler
from vercor.setups import VerosConfig, make_veros_gcm
options = {{}} if {selection!r} == 'default' else {{'backend': 'numpy'}}
ocean = make_veros_gcm(config=VerosConfig(setup='acc', **options))
from veros import runtime_settings
expected = 'jax' if {selection!r} == 'default' else 'numpy'
assert runtime_settings.backend == expected, (runtime_settings.backend, expected)
assert ocean.spec.execution == 'host'
coupler = Coupler(Clock(datetime(2000,1,3),43200.,1), components=(ocean,), run_order=(ocean.name,), log_level='ERROR')
initial = coupler.initial_state()
payload = initial._component_state('OCN').payload
assert isinstance(payload.variables.temp, jax.Array if expected == 'jax' else np.ndarray)
final = coupler.run(initial)
assert int(final._component_state('OCN').payload.variables.itt) == int(payload.variables.itt) + 1
assert int(payload.variables.itt) == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    log = tmp_path / f"native-{selection}.log"
    log.write_text(result.stdout + result.stderr)
    assert (
        result.returncode == 0
    ), f"{(result.stdout + result.stderr)[-2200:]}\nFull log: {log}"
