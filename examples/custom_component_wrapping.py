from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vercor import (
    Component,
    DataComponent,
    HostComponent,
    RectilinearGrid,
    StepContext,
    StepResult,
)
from vercor.dtypes import as_jax_real_array


def make_example_grid() -> RectilinearGrid:
    """Return a small grid for custom component wrapper examples."""

    return RectilinearGrid(
        name="example-grid",
        longitude=as_jax_real_array([0.0, 90.0]),
        latitude=as_jax_real_array([-30.0, 30.0]),
    )


def make_data_forcing(grid: RectilinearGrid) -> DataComponent:
    """Wrap static or time-dependent forcing fields without a runtime step."""

    return DataComponent.from_fields(
        name="ATM",
        grid=grid,
        fields={
            "temperature": 288.15,
            "specific_humidity": 0.01,
        },
    ).update_settings(identifier="example-forcing")


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

    return Component.from_step(
        name="OCN",
        grid=grid,
        step=step,
        inputs=("net_surface_heat_flux",),
        outputs=("sea_surface_temperature",),
        default_fields={
            "sea_surface_temperature": 288.15,
            "net_surface_heat_flux": 0.0,
        },
    )


@dataclass
class ToyHostModel:
    """Small mutable host-side model used to show host wrapper payloads."""

    offset: float = 0.0

    def advance(self, temperature: Any, dt_seconds: float) -> Any:
        self.offset += 0.001 * dt_seconds
        return as_jax_real_array(temperature) + self.offset


def make_host_model(grid: RectilinearGrid) -> HostComponent:
    """Wrap a Python host-side model while keeping VerCOR runtime fields explicit."""

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any | None,
    ) -> StepResult:
        if not isinstance(payload, ToyHostModel):
            raise TypeError("Host wrapper payload must be a ToyHostModel")
        updated_temperature = payload.advance(fields["temperature"], context.dt_seconds)
        return StepResult(
            fields={"temperature": updated_temperature},
            payload=payload,
        )

    return HostComponent.from_step(
        name="LND",
        grid=grid,
        step=step,
        payload=ToyHostModel(),
        outputs=("temperature",),
        default_fields={"temperature": 283.15},
    )


if __name__ == "__main__":
    example_grid = make_example_grid()
    for component in (
        make_data_forcing(example_grid),
        make_differentiable_model(example_grid),
        make_host_model(example_grid),
    ):
        print(component)
