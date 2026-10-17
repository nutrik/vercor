"""Synthetic native lead-closing-thickness recovery with public VerCOR state.

Fit h0 (m) against concentration on fixed wet support, including ice-free cells.
The weighted loss uses a 1e-3 concentration scale and bounded L-BFGS-B with JAX
value/gradient evaluation. This demonstrates numerical identifiability and
parameter recovery, not observational calibration or climate skill. Geometry,
initial conditions, hard physical transitions and numerical controls stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import DataComponent
from vercor.dtypes import DTypePolicy, as_jax_real_array, jax_full
from vercor.recipes import (
    ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
    OCEAN_TO_VERIS_FORCING_FIELDS,
)
from vercor.setups import VerisConfig, make_veris, replace_veris_parameters
from vercor.state import RunState
from vercor.types import RuntimeArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VerisCalibrationResult:
    """Fitted h0, scaled loss/gradient and optimizer counts/convergence flag."""

    h0: jax.Array
    loss: jax.Array
    gradient: jax.Array
    iterations: jax.Array
    evaluations: jax.Array
    converged: jax.Array


@partial(jax.jit, static_argnames=("coupler", "component"))
def _predict_fraction(
    h0: jax.Array, initial: RunState, *, coupler: Coupler, component: str
) -> jax.Array:
    """Predict through public state replacement, fields and Coupler.run only."""
    candidate = replace_veris_parameters(initial, {"h0": h0}, component=component)
    return jnp.asarray(
        coupler.run(candidate).component(component).field("ice_fraction")
    )


@partial(jax.jit, static_argnames=("coupler", "component"))
def _loss_and_gradient(
    h0: jax.Array,
    initial: RunState,
    observed: jax.Array,
    weights: jax.Array,
    *,
    coupler: Coupler,
    component: str,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate a fixed-support residual and its native physical derivative."""

    def loss(value: jax.Array) -> jax.Array:
        predicted = _predict_fraction(
            value, initial, coupler=coupler, component=component
        )
        residual = (predicted - observed) / jnp.asarray(1e-3, dtype=observed.dtype)
        return 0.5 * jnp.sum(weights * residual**2)

    return cast(tuple[jax.Array, jax.Array], jax.value_and_grad(loss)(h0))


def calibrate_lead_closing_thickness(
    coupler: Coupler,
    initial_state: RunState,
    observed_fraction: RuntimeArray,
    *,
    weights: RuntimeArray | None = None,
    initial_h0: float = 0.5,
    bounds: tuple[float, float] = (0.25, 0.8),
    component: str = "CICE",
) -> VerisCalibrationResult:
    """Fit native h0 to a fixed observation window without changing its initial state.

    Observations and optional nonnegative weights have the component grid shape.
    Weights use immutable computational wet support, never evolving ice Area.
    Reject nonfinite/out-of-range observations, invalid/all-dry weights and an
    insensitive window; require positive ordered bounds before optimization. Resolvability
    requires a weighted RMS endpoint span above 32 machine epsilons of the full
    dimensionless concentration range [0, 1]. Counts refer to SciPy iterations
    and objective/gradient evaluations; endpoint probes are separate checks.
    """
    from scipy.optimize import minimize

    grid = initial_state.component(component).grid
    if grid is None:
        raise ValueError("calibration observations require component grid metadata")
    observed_host = np.asarray(observed_fraction)
    if (
        observed_host.shape != grid.shape
        or observed_host.dtype.kind not in "bifu"
        or not np.isfinite(observed_host).all()
        or np.any((observed_host < 0) | (observed_host > 1))
    ):
        raise ValueError(
            "observations must be finite real grid-shaped fractions in [0, 1]"
        )
    weight_host = np.ones(grid.shape) if weights is None else np.asarray(weights)
    if (
        weight_host.shape != grid.shape
        or weight_host.dtype.kind not in "bifu"
        or not np.isfinite(weight_host).all()
        or np.any(weight_host < 0)
    ):
        raise ValueError(
            "weights must be finite nonnegative real values with the grid shape"
        )
    wet = (
        np.ones(grid.shape, dtype=bool)
        if grid.binary_mask is None
        else np.asarray(grid.binary_mask) > 0
    )
    weight_host = np.where(wet, weight_host, 0.0)
    if not np.any(weight_host > 0):
        raise ValueError("weights must include positive wet support")
    authored_bounds = np.asarray(bounds)
    if (
        authored_bounds.shape != (2,)
        or authored_bounds.dtype.kind not in "ifu"
        or not np.isfinite(authored_bounds).all()
        or not 0 < authored_bounds[0] < authored_bounds[1]
    ):
        raise ValueError("bounds must be two finite ordered positive h0 values")
    lower, upper = map(float, authored_bounds)
    if not np.isfinite(initial_h0) or not lower <= initial_h0 <= upper:
        raise ValueError("initial_h0 must be finite and inside bounds")
    policy = coupler.runtime.dtype
    observed = as_jax_real_array(observed_host, policy)
    normalized = as_jax_real_array(weight_host / weight_host.sum(), policy)
    endpoints = [
        np.asarray(
            _predict_fraction(
                as_jax_real_array(value, policy),
                initial_state,
                coupler=coupler,
                component=component,
            )
        )
        for value in (lower, upper)
    ]
    span = np.sqrt(np.sum(np.asarray(normalized) * (endpoints[1] - endpoints[0]) ** 2))
    if not np.isfinite(span) or span <= 32 * np.finfo(policy.numpy_real).eps:
        raise ValueError("observation window is insensitive to h0 within bounds")

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        loss, gradient = _loss_and_gradient(
            as_jax_real_array(values[0], policy),
            initial_state,
            observed,
            normalized,
            coupler=coupler,
            component=component,
        )
        return float(loss), np.asarray([float(gradient)], dtype=float)

    fitted = minimize(
        objective,
        np.asarray([initial_h0]),
        method="L-BFGS-B",
        jac=True,
        bounds=[(lower, upper)],
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 100},
    )
    h0 = as_jax_real_array(fitted.x[0], policy)
    loss, gradient = _loss_and_gradient(
        h0, initial_state, observed, normalized, coupler=coupler, component=component
    )
    return VerisCalibrationResult(
        h0,
        loss,
        gradient,
        jnp.asarray(fitted.nit, dtype=policy.jax_index),
        jnp.asarray(fitted.nfev, dtype=policy.jax_index),
        jnp.asarray(fitted.success),
    )


def _cold_fields(
    shape: tuple[int, int],
    *,
    policy: DTypePolicy,
    holdout: bool = False,
    calm: bool = False,
) -> dict[str, RuntimeArray]:
    """Prescribe smooth cold meteorology at the explicit setup precision."""
    policy._ensure_jax_capability()
    temperature = 258.0 if holdout else 260.0
    wind_u = 0.0 if calm else 6.0 if holdout else 5.0
    wind_v = 0.0 if calm else 1.5 if holdout else 2.0
    values = {
        "temperature": temperature,
        "potential_temperature": temperature,
        "specific_humidity": 0.0008 if holdout else 0.001,
        "density": 1.3,
        "model_level_height": 10.0,
        "surface_pressure": 1e5,
        "u_velocity": wind_u,
        "v_velocity": wind_v,
        "u_velocity_10m": wind_u,
        "v_velocity_10m": wind_v,
        "net_shortwave_radiation_flux": 30.0 if holdout else 20.0,
        "downward_shortwave_radiation_flux": 45.0 if holdout else 30.0,
        "downward_longwave_radiation_flux": 170.0 if holdout else 180.0,
        "sea_surface_temperature": 271.15,
        "sea_surface_salinity": 35.0,
        "ocean_u_velocity": 0.0,
        "ocean_v_velocity": 0.0,
        "sea_surface_height": 0.0,
        "rain_rate": 0.0,
        "snow_rate": 1e-6,
    }
    return {name: jax_full(shape, value, policy) for name, value in values.items()}


def _make_coupler(
    *,
    loglevel: str,
    float_type: str,
    steps: int = 4,
    holdout: bool = False,
    initial_area: float = 0.5,
    calm: bool = False,
) -> Coupler:
    """Build a five-wet-cell planar native recovery experiment without assets."""
    if float_type not in {"float32", "float64"}:
        raise ValueError("float_type must be float32 or float64")
    import veris

    policy = DTypePolicy(enable_x64=float_type == "float64")
    mask = np.ones((3, 2))
    mask[0, 0] = 0.0
    mask_u = mask * np.roll(mask, 1, axis=0)
    mask_v = mask * np.roll(mask, 1, axis=1)
    mask_v[:, 0] = 0.0
    grid = RectilinearGrid(
        "synthetic-ocean",
        longitude=np.arange(3.0),
        latitude=np.asarray([-1.0, 1.0]),
        binary_mask=mask.T,
    )
    geometry = veris.OceanGeometry(
        maskT=mask[..., None],
        maskU=mask_u[..., None],
        maskV=mask_v[..., None],
        ht=-1000 * mask,
        coriolis_t=np.full((3, 2), 1e-4),
        dxt=np.full(3, 2e4),
        dxu=np.full(3, 2e4),
        dyt=np.full(2, 2e4),
        dyu=np.full(2, 2e4),
        area_t=np.full((3, 2), 4e8),
        area_u=np.full((3, 2), 4e8),
        area_v=np.full((3, 2), 4e8),
    )
    ice = make_veris(
        grid,
        geometry,
        config=VerisConfig(
            model_timestep=timedelta(seconds=300),
            custom_parameters={"nEVPsteps": 8},
            initial_conditions={
                "Area": initial_area,
                "hIceMean": 0.5 * initial_area,
                "TSurf": 260.0,
            },
        ),
    )
    source = DataComponent(
        "FORCING",
        grid,
        _cold_fields(grid.shape, policy=policy, holdout=holdout, calm=calm),
    )
    return Coupler(
        Clock(datetime(2000, 1, 1), 600.0, steps),
        components=(source, ice),
        exchanges=(
            Exchange(
                "FORCING",
                "CICE",
                (*ATMOSPHERE_TO_VERIS_FORCING_FIELDS, *OCEAN_TO_VERIS_FORCING_FIELDS),
                quantity="intensive",
            ),
        ),
        run_order=("FORCING", "CICE"),
        runtime=RuntimeOptions(dtype=policy),
        log_level=loglevel,
    )


def run_setup(*, loglevel: str, float_type: str) -> None:
    """Recover synthetic h0=0.43 m and report an independent cold-forcing holdout."""
    coupler = _make_coupler(loglevel=loglevel, float_type=float_type)
    initial = coupler.initial_state()
    truth = as_jax_real_array(0.43, coupler.runtime.dtype)
    observed = _predict_fraction(truth, initial, coupler=coupler, component="CICE")
    fitted = calibrate_lead_closing_thickness(
        coupler, initial, observed, initial_h0=0.3
    )
    holdout = _make_coupler(
        loglevel=loglevel, float_type=float_type, steps=6, holdout=True
    )
    held_initial = holdout.initial_state()
    error = jnp.max(
        jnp.abs(
            _predict_fraction(
                fitted.h0, held_initial, coupler=holdout, component="CICE"
            )
            - _predict_fraction(truth, held_initial, coupler=holdout, component="CICE")
        )
    )
    print(
        f"Synthetic h0: truth=0.43 recovered={float(fitted.h0):.9g} loss={float(fitted.loss):.3g} "
        f"gradient={float(fitted.gradient):.3g} iterations={int(fitted.iterations)} "
        f"evaluations={int(fitted.evaluations)} converged={bool(fitted.converged)} holdout_error={float(error):.3g}"
    )


if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
