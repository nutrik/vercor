"""Actual native ice-albedo feedback through JCM and a later ocean interval."""

from dataclasses import replace
from typing import Any
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vercor import Coupler
from vercor.regridding import conservative
from vercor.setups import replace_veris_parameters
from tests.test_veris_veros_jcm_gallery import native_coupler

pytestmark = [pytest.mark.veros_autodiff, pytest.mark.slow, pytest.mark.fast_always]


class _StoppedSurface:
    """Test-only remapper preserving forward values while removing a derivative path."""

    def __init__(self, source_grid: Any, target_grid: Any):
        self.base = conservative(source_grid, target_grid)
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.has_identical_grids = self.base.has_identical_grids

    def regrid(self, field: Any) -> Any:
        return jax.lax.stop_gradient(self.base.regrid(field))


def test_native_albedo_has_nonzero_atmosphere_mediated_ocean_gradient(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vercor.setups._external.veros_runtime_settings import configure_veros_runtime

    configure_veros_runtime("jax")
    coupler = native_coupler(monkeypatch, execution="jax", steps=3)
    initial = coupler.initial_state()
    stopped = Coupler(
        coupler.clock,
        components=tuple(coupler.components.values()),
        exchanges=tuple(
            (
                replace(route, regridder_factory=_StoppedSurface)
                if route.source == "CICE" and route.target == "ATM"
                else route
            )
            for route in coupler.exchanges
        ),
        run_order=coupler.run_order,
        runtime=coupler.runtime,
        log_level="error",
    )
    stopped.initial_state()
    wet = jnp.asarray(initial.component("OCN").grid.binary_mask)
    names = (
        "dryIceAlb",
        "wetIceAlb",
        "dryIceAlb_south",
        "wetIceAlb_south",
        "drySnowAlb",
        "wetSnowAlb",
        "drySnowAlb_south",
        "wetSnowAlb_south",
    )

    def objective(alpha: jax.Array, engine: Coupler = coupler) -> jax.Array:
        final = engine.run(
            replace_veris_parameters(initial, {name: alpha for name in names})
        )
        # milli-K change keeps the derivative numerically legible.
        temperature = final.component("OCN").field("sea_surface_temperature")
        return 1000 * jnp.sum((temperature - 271.15) * wet) / jnp.sum(wet)

    alpha = jnp.array(0.70, dtype=jnp.float64)
    compiled = jax.jit(objective)
    value, forward = jax.jvp(compiled, (alpha,), (jnp.ones_like(alpha),))
    reverse = jax.grad(compiled)(alpha)
    assert np.isfinite(value) and abs(float(reverse)) > 1e-8
    np.testing.assert_allclose(forward, reverse, rtol=2e-7, atol=1e-9)
    finite = [
        (compiled(alpha + h) - compiled(alpha - h)) / (2 * h)
        for h in (1e-2, 1e-3, 1e-4)
    ]
    np.testing.assert_allclose(finite, reverse, rtol=3e-3, atol=1e-7)

    def control(x: jax.Array) -> jax.Array:
        return objective(x, stopped)

    np.testing.assert_allclose(control(alpha), value, rtol=0, atol=1e-10)
    partial = jax.grad(control)(alpha)
    mediated = reverse - partial
    assert abs(float(mediated)) > 1e-8, (float(reverse), float(partial))

    (tmp_path / "native-feedback-gradient.json").write_text(
        json.dumps(
            {
                "value_mK": float(value),
                "jvp": float(forward),
                "vjp": float(reverse),
                "finite_differences": [float(x) for x in finite],
                "direct_control": float(partial),
                "atmosphere_mediated": float(mediated),
            },
            indent=2,
        )
        + "\n"
    )
