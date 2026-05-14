from __future__ import annotations

from collections.abc import Iterable

from vercor.clock import Clock
from vercor.components.base import Component
from vercor.coupler import Coupler
from vercor.exchange import Exchange
from vercor.run_sequence import RunSequence


def build_coupler(
    *,
    clock: Clock,
    components: Iterable[Component],
    run_sequence: RunSequence,
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
