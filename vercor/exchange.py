from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, TypeAlias

from vercor.regridders.bilinear import BilinearRectilinearRegridder
from vercor.regridders.conservative import ConservativeRectilinearRegridder

ExchangeField: TypeAlias = str | tuple[str, str]
RegridderFactory: TypeAlias = Callable[
    ..., BilinearRectilinearRegridder | ConservativeRectilinearRegridder
]


def _regridder_factory_name(regridder_factory: Callable[..., object]) -> str:
    """Return a stable display name for a regridder factory callable."""

    if isinstance(regridder_factory, partial):
        wrapped_factory = regridder_factory.func
        if callable(wrapped_factory):
            return _regridder_factory_name(wrapped_factory)
        return wrapped_factory.__class__.__name__

    name = getattr(regridder_factory, "__name__", None)
    if isinstance(name, str):
        return name
    return regridder_factory.__class__.__name__


@dataclass
class Exchange:
    """Public exchange declaration connecting source fields to a destination.

    Exchange objects are static configuration. The coupler converts them into
    runtime contracts and dispatch metadata before execution so traced runtime
    state only carries arrays and stable field-store metadata.

    Attributes:
        source, destination: component names
        name: exchange name (automatically setup from source, destination, and regridder_factory)
        field_names: sequence of scalar field names and
            tuples of vectors (u-component, v-component)
        regridder_factory: callable that returns a Regridder instance
        interpolation_type: type of interpolation used (automatically set from regridder_factory)
    """

    source: str
    destination: str
    name: str = field(init=False)
    field_names: Sequence[ExchangeField]
    regridder_factory: RegridderFactory
    interpolation_type: str = field(init=False)

    def __post_init__(self) -> None:
        factory_name = _regridder_factory_name(self.regridder_factory)
        self.name = f"{self.source} --({factory_name})--> {self.destination}"
        self.interpolation_type = factory_name

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Name: {self.name}\n"
            f"├── Source component: {self.source}\n"
            f"└── Destination component: {self.destination}\n"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, source={self.source},"
            f" destination={self.destination}, fields={self.field_names})"
        )
