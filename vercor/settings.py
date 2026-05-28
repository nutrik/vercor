from __future__ import annotations

from typing import Any, NamedTuple

from vercor.dtypes import DTypePolicy
from vercor.physical_constants import PHYSICAL_CONSTANT_SETTINGS


class Settings(NamedTuple):
    """Metadata record for one VerCOR setting."""

    value: Any
    description: str
    units: str


CONTROL_SETTINGS: dict[str, Settings] = {
    # Runtime settings
    "enable_x64": Settings(False, "Enable 64-bit precision for JAX computations", "-"),
    "identifier": Settings("UNNAMED", "Identifier of the current simulation", "-"),
    "missval": Settings(0.0, "Missing value for fields", "-"),
    "apply_time_interpolation": Settings(
        False,
        "Apply monthly time interpolation to exported forcing data",
        "-",
    ),
    "get_field_time_slice": Settings(
        False,
        "Export only the relevant daily time slice from forcing data",
        "-",
    ),
    "year_in_seconds": Settings(365 * 86400.0, "Nominal model year length", "s"),
}


DEFAULT_SETTINGS: dict[str, Settings] = {
    **CONTROL_SETTINGS,
    **{
        name: Settings(value, description, units)
        for name, (value, description, units) in PHYSICAL_CONSTANT_SETTINGS.items()
    },
}


def _copy_settings(settings: dict[str, Settings]) -> dict[str, Settings]:
    """Return independent settings records for a new settings container."""

    return {
        name: Settings(record.value, record.description, record.units)
        for name, record in settings.items()
    }


class VercorSettings:
    """Mutable metadata-backed settings container for couplers and components.

    Known default settings are class-level annotations for static type checkers;
    runtime values live in ``_settings`` and are resolved dynamically.
    """

    _settings: dict[str, Settings]
    enable_x64: bool
    identifier: str
    missval: float
    apply_time_interpolation: bool
    get_field_time_slice: bool
    year_in_seconds: float
    earth_radius: float
    gravity: float
    rhoAir: float
    rdair: float
    cpdair: float
    zvir: float
    p0: float
    mwdair: float
    cpwv: float
    cpvir: float
    cappa: float
    latice: float
    rgas: float
    umin_ocean: float
    umin_ice: float
    karman: float
    stefBoltz: float
    ocean_emissivity: float
    ice_emissivity: float
    snow_emissivity: float
    latvap: float
    latfresh: float
    gamma_blk: float
    zref: float
    ztref: float

    def __init__(self, **kwargs: Any) -> None:
        """Create settings from VerCOR defaults plus optional overrides."""

        object.__setattr__(self, "_settings", _copy_settings(DEFAULT_SETTINGS))
        for name, value in kwargs.items():
            if isinstance(value, Settings):
                self._settings[name] = Settings(
                    value.value,
                    value.description,
                    value.units,
                )
            elif name in self._settings:
                self.set_value(name, value)
            else:
                self.add_setting(name, value)

    def __getattr__(self, name: str) -> Any:
        """Return the value of a setting by attribute name."""

        settings = object.__getattribute__(self, "_settings")
        if name in settings:
            return settings[name].value
        raise AttributeError(
            f"{self.__class__.__name__!s} has no setting named {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Update an existing setting value through attribute assignment."""

        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        settings = object.__getattribute__(self, "_settings")
        if name not in settings:
            raise AttributeError(
                f"{self.__class__.__name__!s} has no setting named {name!r}; "
                "use add_setting() to add custom settings"
            )
        self.set_value(name, value)

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is a configured setting."""

        return name in self._settings

    def __dir__(self) -> list[str]:
        """Return normal instance attributes plus configured setting names."""

        names = set(super().__dir__())
        names.update(self._settings)
        return sorted(names)

    def __repr__(self) -> str:
        values = ", ".join(
            f"{name}={record.value!r}" for name, record in self._settings.items()
        )
        return f"{self.__class__.__name__}({values})"

    def add_setting(
        self,
        name: str,
        value: Any,
        description: str = "-",
        units: str = "-",
    ) -> None:
        """Add a custom setting to this container."""

        if name in self._settings:
            raise KeyError(f"Setting {name!r} already exists")
        if isinstance(value, Settings):
            self._settings[name] = Settings(value.value, value.description, value.units)
            return
        self._settings[name] = Settings(value, description, units)

    def set_value(self, name: str, value: Any) -> None:
        """Update the value of an existing setting while preserving metadata."""

        if name not in self._settings:
            raise KeyError(f"Setting {name!r} does not exist")
        metadata = self._settings[name]
        if isinstance(value, Settings):
            self._settings[name] = Settings(value.value, value.description, value.units)
            return
        self._settings[name] = Settings(value, metadata.description, metadata.units)

    def get_value(self, name: str) -> Any:
        """Return a setting value by name."""

        if name not in self._settings:
            raise KeyError(f"Setting {name!r} does not exist")
        return self._settings[name].value

    def get_metadata(self, name: str) -> Settings:
        """Return the full metadata record for one setting."""

        if name not in self._settings:
            raise KeyError(f"Setting {name!r} does not exist")
        record = self._settings[name]
        return Settings(record.value, record.description, record.units)

    def as_values(self) -> dict[str, Any]:
        """Return a plain mapping of setting names to values."""

        return {name: record.value for name, record in self._settings.items()}

    @property
    def dtype_policy(self) -> DTypePolicy:
        """Return the canonical array dtype policy for these settings."""

        return DTypePolicy.from_settings(self)
