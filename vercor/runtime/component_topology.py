from __future__ import annotations

from typing import TYPE_CHECKING

from vercor.exceptions import ComponentError, CouplerError

if TYPE_CHECKING:
    from vercor.components.base import Component


VALID_TOPOLOGY_COMPONENT_NAMES = ("ATM", "OCN", "LND", "ICE")


def validate_component_topology_names(components: dict[str, Component]) -> None:
    """Validate registered component names supported by the default topology."""

    for name in components:
        if name not in VALID_TOPOLOGY_COMPONENT_NAMES:
            allowed = ", ".join(VALID_TOPOLOGY_COMPONENT_NAMES)
            raise ComponentError(f"Incorrect component name: {name}, must be {allowed}")


def get_component(allcomponents: dict[str, Component], types: str) -> Component:
    """Return the registered component with the requested VerCOR component name."""

    components: list[Component] = [
        component for component in allcomponents.values() if component.name == types
    ]
    if len(components) > 1:
        raise CouplerError(
            f"Multiple {components[0].name} components registered; only one supported"
        )
    if not components:
        raise CouplerError(f"No component of types ({types}) registered")
    return components[0]
