from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    Component,
    DataComponent,
    ComponentSpec,
    LifecycleHooks,
    SetupResult,
    StepContext,
    StepResult,
)
from vercor.coupler import Coupler
from vercor.exchanges import Exchange
from vercor.grids import RectilinearGrid
from vercor.runtime import (
    ExecutionChunk,
    ExecutionContext,
    RuntimeDriver,
    RuntimeOptions,
)
from vercor.state import RunState
from vercor.dtypes import as_jax_real_array


def make_example_grid() -> RectilinearGrid:
    """Return a small grid for custom component wrapper examples."""

    return RectilinearGrid.from_coordinates(
        "example-grid",
        longitude=as_jax_real_array([0.0, 90.0]),
        latitude=as_jax_real_array([-30.0, 30.0]),
    )


def make_data_forcing(grid: RectilinearGrid) -> DataComponent:
    """Wrap static or time-dependent forcing fields without a runtime step."""

    return DataComponent(
        "ATM",
        grid,
        {
            "temperature": 288.15,
            "specific_humidity": 0.01,
        },
    )


def make_differentiable_model(grid: RectilinearGrid) -> Component:
    """Wrap a pure JAX callable as a differentiable VerCOR component."""

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        heat_capacity = 1025.0 * 3990.0 * 30.0
        tendency = fields["net_surface_heat_flux"] / heat_capacity
        return {
            "sea_surface_temperature": (
                fields["sea_surface_temperature"] + tendency * context.dt_seconds
            )
        }

    return CallableComponent(
        "OCN",
        grid,
        step,
        spec=ComponentSpec(
            inputs=("net_surface_heat_flux",),
            outputs=("sea_surface_temperature",),
            initial_fields={
                "sea_surface_temperature": 288.15,
                "net_surface_heat_flux": 0.0,
            },
        ),
    )


@dataclass(frozen=True)
class ToyHostModel:
    """Small functional host-side model used to show host wrapper payloads."""

    offset: float = 0.0

    def advance(
        self, temperature: Any, dt_seconds: float
    ) -> tuple[Any, "ToyHostModel"]:
        """Return the updated field and a replacement immutable payload."""

        next_model = ToyHostModel(self.offset + 0.001 * dt_seconds)
        return as_jax_real_array(temperature) + next_model.offset, next_model


def make_host_model(grid: RectilinearGrid) -> Component:
    """Wrap a Python host-side model while keeping VerCOR runtime fields explicit."""

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any | None,
    ) -> StepResult:
        if not isinstance(payload, ToyHostModel):
            raise TypeError("Host wrapper payload must be a ToyHostModel")
        updated_temperature, next_payload = payload.advance(
            fields["temperature"], context.dt_seconds
        )
        return StepResult(
            fields={"temperature": updated_temperature},
            payload=next_payload,
        )

    return CallableComponent(
        "LND",
        grid,
        step,
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 283.15},
            execution="host",
            lifecycle=LifecycleHooks(
                setup=lambda component, context: SetupResult(payload=ToyHostModel())
            ),
        ),
    )


@dataclass
class StructuralFluxModel:
    """Small structural component using the public Component protocol."""

    grid: RectilinearGrid
    name: str = "MODEL"
    spec: ComponentSpec = ComponentSpec(
        inputs=("custom_flux",),
        outputs=("custom_flux",),
        initial_fields={"custom_flux": 0.0},
        execution="host",
    )

    def step(
        self,
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, Any]:
        """Update the custom flux on the host runtime path."""

        _ = payload
        return {"custom_flux": fields["custom_flux"] + context.step}


class SequentialBackend:
    """Minimal custom backend that delegates component stepping to RuntimeDriver."""

    def execute(
        self,
        state: RunState,
        *,
        context: ExecutionContext,
        chunk: ExecutionChunk,
        driver: RuntimeDriver,
    ) -> RunState:
        """Run every step plan in one core-defined chunk."""

        _ = context
        for plan in chunk.steps:
            state = driver.run_step(state, plan)
        return state


def make_custom_coupler(grid: RectilinearGrid) -> Coupler:
    """Assemble custom-named components without the built-in surface-mask policy."""

    source = DataComponent("FORCING", grid, {"custom_flux": 1.0})

    model = StructuralFluxModel(grid)
    return Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=3),
        components=(source, model),
        exchanges=(Exchange("FORCING", "MODEL", ("custom_flux",)),),
        run_order=("FORCING", "MODEL"),
        runtime=RuntimeOptions(backend=SequentialBackend()),
    )


if __name__ == "__main__":
    example_grid = make_example_grid()
    for component in (
        make_data_forcing(example_grid),
        make_differentiable_model(example_grid),
        make_host_model(example_grid),
    ):
        print(component)
    print(make_custom_coupler(example_grid))
