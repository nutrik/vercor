from __future__ import annotations

from collections.abc import Iterable, Sequence

from vercor._deprecation import deprecated_getattr, warn_deprecated_name
from vercor.clock import Clock
from vercor.components import Component
from vercor.coupler import Coupler
from vercor.exchange import Exchange


def build_coupler(
    *,
    clock: Clock,
    components: Iterable[Component],
    run_sequence: Sequence[str],
) -> Coupler:
    """Build a coupler with components registered and run order configured."""

    warn_deprecated_name(
        "vercor.setups.build_coupler()",
        "Coupler.from_components()",
        remove_in="0.2.0",
    )
    return Coupler.from_components(
        clock=clock,
        components=components,
        run_order=run_sequence,
    )


def add_exchanges(coupler: Coupler, exchanges: Iterable[Exchange]) -> None:
    """Add a sequence of explicit exchange declarations to a coupler."""

    warn_deprecated_name(
        "vercor.setups.add_exchanges()",
        "Coupler.add_exchanges()",
        remove_in="0.2.0",
    )
    for exchange in exchanges:
        coupler.add_exchange(exchange)


def build_exchanges(specs: Iterable[Exchange]) -> tuple[Exchange, ...]:
    """Build public exchange declarations from compact setup specs."""

    warn_deprecated_name(
        "vercor.setups.build_exchanges()",
        "tuple(exchanges)",
        remove_in="0.2.0",
    )
    return tuple(specs)


def add_exchange_specs(coupler: Coupler, specs: Iterable[Exchange]) -> None:
    """Build and add exchange declarations from compact setup specs."""

    warn_deprecated_name(
        "vercor.setups.add_exchange_specs()",
        "Coupler.add_exchanges()",
        remove_in="0.2.0",
    )
    add_exchanges(coupler, build_exchanges(specs))


__getattr__ = deprecated_getattr(
    __name__,
    {
        "ExchangeSpec": ("vercor.exchange.Exchange", Exchange),
    },
    remove_in="0.2.0",
)
