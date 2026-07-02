from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vercor.exceptions import ComponentError, CouplerError

if TYPE_CHECKING:
    from vercor.components.base import Component


VALID_TOPOLOGY_COMPONENT_NAMES = ("ATM", "OCN", "LND", "ICE")


def validate_component_topology_names(components: Mapping[str, Component]) -> None:
    """Validate registered component names supported by the default topology."""

    for name in components:
        if name not in VALID_TOPOLOGY_COMPONENT_NAMES:
            allowed = ", ".join(VALID_TOPOLOGY_COMPONENT_NAMES)
            raise ComponentError(f"Incorrect component name: {name}, must be {allowed}")


def require_component(components: Mapping[str, Component], name: str) -> Component:
    """Return the component registered under ``name`` or raise a coupler error."""

    try:
        component = components[name]
    except KeyError as exc:
        raise CouplerError(f"No component of type {name!r} registered") from exc

    if component.name != name:
        raise CouplerError(
            f"Component registered under key {name!r} has name {component.name!r}; "
            "component mapping keys must match component.name"
        )
    return component
