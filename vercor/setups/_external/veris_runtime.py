"""Native Veris initialization and pure coupled substeps with payload-owned state.

Host setup binds immutable native numerical configuration and an aligned integer
substep count. The traced payload holds all evolving native State/Diagnostics
arrays, including halos. Forcing is packed once per coupling interval and
reapplied before EVERY native substep because Growth replaces Qnet/Qsw with
residuals. No native Simulation, clock, output writer or mutable model is stepped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields as dataclass_fields, replace
from typing import Any, TypeAlias, cast

import jax
import jax.numpy as jnp

from vercor.components import (
    Component,
    SetupContext,
    SetupResult,
    StepContext,
    StepResult,
)
from vercor.dtypes import DTypePolicy
from vercor.exceptions import ComponentError
from vercor._numerical_safety import _require_zero
from vercor.setups.config import VerisConfig
from vercor.setups._time_helpers import align_model_timestep
from vercor.setups._external.veris_forcing import prepare_veris_forcing_pack
from vercor.setups._external.veris_fluxes import (
    VerisSurfaceFluxes,
    VerisIntervalBudget,
    veris_surface_fluxes,
)
from vercor.types import RuntimeArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VerisPayload:
    """Complete evolving native state plus same-schema per-step diagnostics."""

    state: Any
    diagnostics: Any
    physical_constants: Any
    interval_budget: VerisIntervalBudget


_VerisCarry: TypeAlias = tuple[
    VerisPayload, VerisSurfaceFluxes, VerisIntervalBudget, jax.Array
]


@dataclass(frozen=True)
class VerisRuntimeResources:
    """Static numerical configuration and coupling alignment, with no physical values."""

    configuration: Any
    model_substeps: int
    dtype: DTypePolicy


def _runtime_veris_constants(constants: Any) -> Any:
    """Normalize independent leaves to the native runtime precision schema."""
    return jax.tree.map(
        lambda value: jnp.asarray(value, dtype=constants.dtype), constants
    )


@jax.jit
def _finite_leaves(value: Any) -> jax.Array:
    """Reduce finiteness without issuing per-leaf host callbacks."""
    valid = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(value):
        valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


@jax.jit
def _valid_native_payload(payload: VerisPayload) -> jax.Array:
    """Reduce all owned numeric leaves, native bounds and status to one flag."""
    valid = _finite_leaves(payload)
    state = payload.state
    return cast(
        jax.Array,
        (
            valid
            & jnp.all(state.hIceMean >= 0)
            & jnp.all(state.hSnowMean >= 0)
            & jnp.all((state.Area >= 0) & (state.Area <= 1))
            & jnp.all(payload.diagnostics.physics_status == 0)
        ),
    )


def step_veris_runtime(
    resources: VerisRuntimeResources,
    fields: Mapping[str, RuntimeArray],
    context: StepContext,
    payload: VerisPayload,
) -> StepResult:
    """Integrate complete boundary rates and reject invalid intermediate physics.

    All native inputs and outputs are checked each substep. A failed carry
    skips later physics; one aggregate guard at the coupling boundary prevents
    accepting it, including under outer JIT. Final diagnostics stay instantaneous
    while the five ordinary exported fields and external budget are interval means.
    """
    if not isinstance(payload, VerisPayload):
        raise ComponentError("Veris requires its initialized native runtime payload")
    from veris import advance

    forcing = prepare_veris_forcing_pack(
        fields,
        ocean_mask=payload.state.iceMask[2:-2, 2:-2].T,
        constants=context.constants,
        dtype=context.dtype,
    )
    resolved = _runtime_veris_constants(
        payload.physical_constants.with_overrides(
            rhoFresh=context.constants.freshwater_density
        )
    )
    payload = replace(payload, physical_constants=resolved)
    zero = jnp.zeros_like(payload.state.Area[2:-2, 2:-2].T)
    totals = VerisSurfaceFluxes(zero, zero, zero, zero, zero)
    external = VerisIntervalBudget(zero, zero)
    dt = jnp.asarray(resources.configuration.deltatTherm, dtype=context.dtype.jax_real)

    def substep(
        _: int,
        carry: _VerisCarry,
    ) -> _VerisCarry:
        current, total, boundary, valid = carry
        valid = valid & _valid_native_payload(current)

        def advance_valid(
            _: None,
        ) -> _VerisCarry:
            state, diagnostics = advance(
                current.state,
                resources.configuration,
                current.physical_constants,
                forcing.native_forcing,
                physics="coupled",
            )
            updated = replace(current, state=state, diagnostics=diagnostics)
            fluxes = veris_surface_fluxes(
                state, diagnostics, forcing, dtype=context.dtype
            )
            wet = state.iceMask[2:-2, 2:-2].T > 0
            inputs = VerisIntervalBudget(
                jnp.where(wet, diagnostics.external_mass_flux[2:-2, 2:-2].T, 0.0),
                jnp.where(wet, diagnostics.external_heat_flux[2:-2, 2:-2].T, 0.0),
            )
            return (
                updated,
                jax.tree.map(
                    lambda accumulated, flux: accumulated + dt * flux, total, fluxes
                ),
                jax.tree.map(
                    lambda accumulated, flux: accumulated + dt * flux, boundary, inputs
                ),
                _valid_native_payload(updated),
            )

        return cast(
            _VerisCarry,
            jax.lax.cond(
                valid,
                advance_valid,
                lambda _: (current, total, boundary, valid),
                None,
            ),
        )

    updated, totals, external, valid = jax.lax.fori_loop(
        0,
        resources.model_substeps,
        substep,
        (payload, totals, external, jnp.asarray(True)),
    )
    duration = jnp.asarray(context.dt_seconds, dtype=context.dtype.jax_real)
    means = jax.tree.map(lambda value: value / duration, totals)
    updated = replace(
        updated, interval_budget=jax.tree.map(lambda value: value / duration, external)
    )
    valid = valid & _finite_leaves((means, updated.interval_budget))
    _require_zero(
        jnp.count_nonzero(~valid),
        "Veris",
        "valid native substeps, interval means and physical status",
    )
    return StepResult(
        fields={"ice_fraction": updated.state.Area[2:-2, 2:-2].T, **means.to_fields()},
        payload=updated,
    )


class VerisSetupState:
    """Setup-only owner; stepping reads frozen resources and replaces payloads."""

    def __init__(self, config: VerisConfig, geometry: Any) -> None:
        """Retain copied host inputs, with no native state before lifecycle setup."""
        self.config = config
        self.geometry = geometry
        self.resources: VerisRuntimeResources | None = None

    def setup(self, component: Component, context: SetupContext) -> SetupResult:
        """Initialize physical geometry at runtime precision with serial native execution."""
        import veris as native
        from veris.model.diagnostics import Diagnostics

        if context.dt_seconds <= 0:
            raise ValueError("coupling_timestep must be positive")
        alignment = align_model_timestep(context.dt_seconds, self.config.model_timestep)
        seconds = alignment.model_timestep.total_seconds()
        dtype_name = "float64" if context.dtype.enable_x64 else "float32"
        owned = {
            "dtype": dtype_name,
            "nx": component.grid.shape[1],
            "ny": component.grid.shape[0],
            "use_sharding": False,
            "enable_cyclic_y": False,
            "deltatDyn": seconds,
            "deltatTherm": seconds,
        }
        settings = dict(self.config.custom_parameters or {})
        for name, value in owned.items():
            if name in settings and settings.pop(name) != value:
                raise ValueError(
                    f"Veris parameter '{name}' is owned by the runtime and conflicts with {value!r}"
                )
        settings.update(
            {
                name: owned[name]
                for name in ("dtype", "enable_cyclic_y", "deltatDyn", "deltatTherm")
            }
        )
        previous = self.resources
        if previous is not None and (
            previous.dtype != context.dtype
            or previous.model_substeps != alignment.model_substeps
            or previous.configuration.deltatDyn != seconds
        ):
            raise ComponentError(
                "Construct a new Veris component for a different runtime dtype or timestep"
            )
        experiment = native.Experiment(
            grid=native.Grid(*component.grid.shape[::-1]),
            geometry=self.geometry,
            settings=settings,
            initial_conditions=self.config.initial_conditions or {},
            constants=self.config.physical_constants or {},
            execution=native.Execution(),
            output=None,
            physics="coupled",
        )
        state, configuration, constants = native.initialize_experiment(experiment)
        if previous is None:
            self.resources = VerisRuntimeResources(
                configuration, alignment.model_substeps, context.dtype
            )
        if "rhoFresh" in (self.config.physical_constants or {}):
            _require_zero(
                jnp.count_nonzero(
                    jnp.asarray(constants.rhoFresh, dtype=context.dtype.jax_real)
                    != context.constants.freshwater_density
                ),
                "Veris rhoFresh",
                "ownership by VerCOR freshwater_density",
            )
        # Native ocean initialization has no rhoFresh-dependent state fields.
        # Apply the traced owner after host definition/initialization, preserving
        # device-free definitions and outer-JIT setup with traced constants.
        constants = _runtime_veris_constants(
            constants.with_overrides(rhoFresh=context.constants.freshwater_density)
        )
        zeros = jnp.zeros_like(state.Area)
        diagnostics = Diagnostics(
            **{field.name: zeros for field in dataclass_fields(Diagnostics)}
        )
        zero = zeros[2:-2, 2:-2].T
        fluxes = VerisSurfaceFluxes(zero, zero, zero, zero, zero)
        return SetupResult(
            fields={"ice_fraction": state.Area[2:-2, 2:-2].T, **fluxes.to_fields()},
            payload=VerisPayload(
                state, diagnostics, constants, VerisIntervalBudget(zero, zero)
            ),
        )

    def step(
        self, fields: Mapping[str, RuntimeArray], context: StepContext, payload: Any
    ) -> StepResult:
        """Delegate to the pure transition without assigning any evolving owner state."""
        if self.resources is None:
            raise ComponentError("Veris must be initialized before stepping")
        return step_veris_runtime(self.resources, fields, context, payload)
