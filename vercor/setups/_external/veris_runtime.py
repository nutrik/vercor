"""Native Veris initialization and pure coupled substeps with payload-owned state.

Host setup binds immutable native configuration/constants and an aligned integer
substep count. The traced payload holds all evolving native State/Diagnostics
arrays, including halos. Forcing is packed once per coupling interval and
reapplied before EVERY native substep because Growth replaces Qnet/Qsw with
residuals. No native Simulation, clock, output writer or mutable model is stepped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any

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
from vercor.setups.config import VerisConfig
from vercor.setups._time_helpers import align_model_timestep
from vercor.setups._external.veris_forcing import prepare_veris_forcing
from vercor.types import RuntimeArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VerisPayload:
    """Complete evolving native state plus same-schema per-step diagnostics."""

    state: Any
    diagnostics: Any


@dataclass(frozen=True)
class VerisRuntimeResources:
    """Static native physics/configuration and coupling alignment, with no state."""

    configuration: Any
    constants: Any
    model_substeps: int
    dtype: DTypePolicy


def step_veris_runtime(
    resources: VerisRuntimeResources,
    fields: Mapping[str, RuntimeArray],
    context: StepContext,
    payload: VerisPayload,
) -> StepResult:
    """Advance coupled dynamics/thermodynamics with freshly prescribed forcing."""
    if not isinstance(payload, VerisPayload):
        raise ComponentError("Veris requires its initialized native runtime payload")
    from veris import advance

    forcing = prepare_veris_forcing(
        fields,
        ocean_mask=payload.state.iceMask[2:-2, 2:-2].T,
        constants=context.constants,
        dtype=context.dtype,
    )

    def substep(_: int, current: VerisPayload) -> VerisPayload:
        state, diagnostics = advance(
            current.state,
            resources.configuration,
            resources.constants,
            forcing,
            physics="coupled",
        )
        return VerisPayload(state, diagnostics)

    updated = jax.lax.fori_loop(0, resources.model_substeps, substep, payload)
    return StepResult(
        fields={"ice_fraction": updated.state.Area[2:-2, 2:-2].T}, payload=updated
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
            execution=native.Execution(),
            output=None,
            physics="coupled",
        )
        state, configuration, constants = native.initialize_experiment(experiment)
        if previous is None:
            self.resources = VerisRuntimeResources(
                configuration, constants, alignment.model_substeps, context.dtype
            )
        zeros = jnp.zeros_like(state.Area)
        diagnostics = Diagnostics(
            **{field.name: zeros for field in dataclass_fields(Diagnostics)}
        )
        return SetupResult(
            fields={"ice_fraction": state.Area[2:-2, 2:-2].T},
            payload=VerisPayload(state, diagnostics),
        )

    def step(
        self, fields: Mapping[str, RuntimeArray], context: StepContext, payload: Any
    ) -> StepResult:
        """Delegate to the pure transition without assigning any evolving owner state."""
        if self.resources is None:
            raise ComponentError("Veris must be initialized before stepping")
        return step_veris_runtime(self.resources, fields, context, payload)
