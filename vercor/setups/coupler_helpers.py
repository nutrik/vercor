from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vercor.clock import Clock
from vercor.components import Component
from vercor.coupler import Coupler
from vercor.exchange import Exchange
from vercor.exchange import ExchangeField, RegridderFactory


@dataclass(frozen=True)
class ExchangeSpec:
    """Compact setup recipe for constructing a public exchange declaration."""

    source: str
    destination: str
    field_names: Sequence[ExchangeField]
    regridder_factory: RegridderFactory


def build_coupler(
    *,
    clock: Clock,
    components: Iterable[Component],
    run_sequence: Sequence[str],
) -> Coupler:
    """Build a coupler with components registered and run order configured."""

    coupler = Coupler(clock=clock)
    for component in components:
        coupler.register(component)
    coupler.set_components_run_sequence(run_sequence)
    return coupler


def add_exchanges(coupler: Coupler, exchanges: Iterable[Exchange]) -> None:
    """Add a sequence of explicit exchange declarations to a coupler."""

    for exchange in exchanges:
        coupler.add_exchange(exchange)


def build_exchanges(specs: Iterable[ExchangeSpec]) -> tuple[Exchange, ...]:
    """Build public exchange declarations from compact setup specs."""

    return tuple(
        Exchange(
            source=spec.source,
            destination=spec.destination,
            field_names=spec.field_names,
            regridder_factory=spec.regridder_factory,
        )
        for spec in specs
    )


def add_exchange_specs(coupler: Coupler, specs: Iterable[ExchangeSpec]) -> None:
    """Build and add exchange declarations from compact setup specs."""

    add_exchanges(coupler, build_exchanges(specs))
