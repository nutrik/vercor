from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Callable, TypeAlias

from vercor._deprecation import warn_deprecated_name
from vercor.regridders.bilinear import BilinearRectilinearRegridder, bilinear
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


@dataclass(frozen=True, init=False)
class Exchange:
    """Public exchange declaration connecting source fields to a destination.

    Exchange objects are static configuration. The coupler converts them into
    runtime contracts and dispatch metadata before execution so traced runtime
    state only carries arrays and stable field-store metadata.
    """

    source: str
    target: str
    fields: Sequence[ExchangeField]
    regrid: RegridderFactory
    name: str | None
    interpolation_type: str

    def __init__(
        self,
        source: str,
        target: str | None = None,
        fields: Sequence[ExchangeField] | None = None,
        regrid: RegridderFactory = bilinear,
        name: str | None = None,
        *,
        destination: str | None = None,
        field_names: Sequence[ExchangeField] | None = None,
        regridder_factory: RegridderFactory | None = None,
    ) -> None:
        """Create an exchange declaration.

        ``destination``, ``field_names``, and ``regridder_factory`` are legacy
        names kept as migration aliases for ``target``, ``fields``, and
        ``regrid``.
        """

        if target is not None and destination is not None:
            raise TypeError("Use either target or destination, not both")
        if fields is not None and field_names is not None:
            raise TypeError("Use either fields or field_names, not both")
        if regridder_factory is not None and regrid is not bilinear:
            raise TypeError("Use either regrid or regridder_factory, not both")
        if destination is not None:
            warn_deprecated_name("destination", "target", remove_in="0.2.0")
        if field_names is not None:
            warn_deprecated_name("field_names", "fields", remove_in="0.2.0")
        if regridder_factory is not None:
            warn_deprecated_name("regridder_factory", "regrid", remove_in="0.2.0")

        resolved_target = target if target is not None else destination
        resolved_fields = fields if fields is not None else field_names
        resolved_regrid = regridder_factory or regrid
        if resolved_target is None:
            raise TypeError("Exchange target is required")
        if resolved_fields is None:
            raise TypeError("Exchange fields are required")

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", resolved_target)
        object.__setattr__(self, "fields", tuple(resolved_fields))
        object.__setattr__(self, "regrid", resolved_regrid)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "interpolation_type",
            _regridder_factory_name(resolved_regrid),
        )

    @property
    def destination(self) -> str:
        """Return legacy destination component name."""

        warn_deprecated_name(
            "Exchange.destination",
            "Exchange.target",
            remove_in="0.2.0",
        )
        return self.target

    @property
    def field_names(self) -> Sequence[ExchangeField]:
        """Return legacy exchange field declaration."""

        warn_deprecated_name(
            "Exchange.field_names",
            "Exchange.fields",
            remove_in="0.2.0",
        )
        return self.fields

    @property
    def regridder_factory(self) -> RegridderFactory:
        """Return legacy regridder factory."""

        warn_deprecated_name(
            "Exchange.regridder_factory",
            "Exchange.regrid",
            remove_in="0.2.0",
        )
        return self.regrid

    @property
    def label(self) -> str:
        """Return explicit name or a stable derived logging label."""

        if self.name is not None:
            return self.name
        return f"{self.source} --({self.interpolation_type})--> {self.target}"

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Name: {self.label}\n"
            f"├── Source component: {self.source}\n"
            f"└── Target component: {self.target}\n"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, source={self.source},"
            f" target={self.target}, fields={self.fields})"
        )
