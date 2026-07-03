from __future__ import annotations

from collections.abc import Iterable, Sequence

from vercor.clock import Clock
from vercor.components import Component
from vercor.coupler import Coupler
from vercor.exchange import Exchange

ExchangeSpec = Exchange
"""Deprecated compatibility alias for :class:`vercor.exchange.Exchange`."""


def build_coupler(
    *,
    clock: Clock,
    components: Iterable[Component],
    run_sequence: Sequence[str],
) -> Coupler:
    """Build a coupler with components registered and run order configured."""

    return Coupler.from_components(
        clock=clock,
        components=components,
        run_order=run_sequence,
    )


def add_exchanges(coupler: Coupler, exchanges: Iterable[Exchange]) -> None:
    """Add a sequence of explicit exchange declarations to a coupler."""

    for exchange in exchanges:
        coupler.add_exchange(exchange)


def build_exchanges(specs: Iterable[ExchangeSpec]) -> tuple[Exchange, ...]:
    """Build public exchange declarations from compact setup specs."""

    return tuple(specs)


def add_exchange_specs(coupler: Coupler, specs: Iterable[ExchangeSpec]) -> None:
    """Build and add exchange declarations from compact setup specs."""

    add_exchanges(coupler, build_exchanges(specs))
