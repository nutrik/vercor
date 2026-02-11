import abc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import h5netcdf
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from vercor.exceptions import ComponentError
from vercor.grid import RectilinearGrid
from vercor.tools import get_field_at_specific_time, get_field_time_slice
from vercor.exchange import VALID_EXCHANGE_FIELD_NAMES


if TYPE_CHECKING:
    from vercor.coupler import Coupler


@dataclass
class TimedNamedArray:
    """Container class for a field (array), its timestamp, and its component name."""

    data: NDArray
    timestamp: datetime
    component_name: str

    def __array__(self, dtype: Optional[NDArray] = None) -> NDArray:
        """Let NumPy see this as an array transparently."""
        return np.asarray(self.data, dtype=dtype)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Component name: {self.component_name!r}\n"
            f"├── Shape: {self.data.shape}\n"
            f"└── Timestamp: {self.timestamp!r}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(component_name={self.component_name!r}, "
            f"shape={self.data.shape}, timestamp={self.timestamp!r})"
        )


@dataclass
class Shared:
    _fields: dict[str, TimedNamedArray] = field(default_factory=dict, init=False)

    def _assign_field(self, name: str, value: Any) -> None:
        # internal attributes
        if name.startswith("_"):
            return super().__setattr__(name, value)

        if isinstance(value, TimedNamedArray):
            self._fields[name] = value
            return

        if isinstance(value, tuple):
            if len(value) == 3:
                data, timestamp, component_name = value
            else:
                raise ValueError(
                    f"Expected tuple of length 3 for field assignment, got length {len(value)}"
                )

            if not isinstance(timestamp, datetime):
                raise TypeError(
                    f"When assigning a tuple, the second element must be a datetime, got {type(timestamp)}"
                )

        else:
            raise TypeError(
                "When assigning a field, provide a tuple (data, timestamp, component name)"
            )

        data = np.asarray(data)
        self._fields[name] = TimedNamedArray(
            data=data,
            timestamp=timestamp,
            component_name=component_name,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        self._assign_field(name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        self._assign_field(name, value)

    def __getattr__(self, name: str) -> TimedNamedArray:
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __getitem__(self, name: str) -> TimedNamedArray | None:
        try:
            return self._fields[name]
        except KeyError:
            print(f"{type(self).__name__!s} has no item {name!r}")
            return None

    def __str__(self) -> str:
        field_descriptions = ", ".join(
            f"{name}({value.component_name})" for name, value in self._fields.items()
        )
        return (
            f"{self.__class__.__name__}:\n"
            f"└── Fields: {field_descriptions if field_descriptions else 'No fields assigned'}"
        )

    def __repr__(self) -> str:
        field_reprs = ", ".join(
            f"{name}={repr(value)}" for name, value in self._fields.items()
        )
        return f"{self.__class__.__name__}({field_reprs})"

    @property
    def is_empty(self) -> bool:
        """Check if the Shared object has no fields."""
        return len(self._fields) == 0

    @property
    def field_names(self) -> list[str]:
        """Return a list of all field names in the Shared object."""
        return list(self._fields.keys())

    def fields(self) -> dict[str, NDArray]:
        """Return a dictionary of all fields' data arrays."""
        return {k: v.data for k, v in self._fields.items()}

    def timestamps(self) -> dict[str, datetime]:
        """Return a dictionary of all fields' timestamps."""
        return {k: v.timestamp for k, v in self._fields.items()}

    def component_names(self) -> dict[str, str]:
        """Return a dictionary of all fields' component names."""
        return {k: v.component_name for k, v in self._fields.items()}


@dataclass
class Component(abc.ABC):
    name: str
    grid: RectilinearGrid
    incoming_fields: Shared = field(default_factory=Shared)
    outgoing_fields: Shared = field(default_factory=Shared)
    data: dict[str, NDArray] = field(default_factory=dict)
    _fields2import: list[str] = field(default_factory=list)
    _fields2export: list[str] = field(default_factory=list)
    _settings: dict[str, Any] = field(default_factory=dict)
    """A component's default grid dimensions are (nTime, nLev, nLon, nLat)

    Some components may have different dimensions, e.g., sea-ice (nTime, nLon, nLat) or
    JCM atmospheric model (nTime, nLev, nLon, nLat).

    One must implement necessary dimensions check and reshaping of fields
    during import/export if needed.

    Common conventions for exchange fields:
        - All fields must have SI units.
        - Surface fluxes are positive downward and negative upward.

    Attributes:
        name: component name
        grid: component grid
        incoming_fields: shared fields received by the current component
                         from another component(s)
        outgoing_fields: shared fields to be sent from the current component
                         to another component(s)
        data: internal storage for component data arrays to/from which fields
                        are imported/exported
        _settings: component-specific settings
        _fields2import: list of field names to import from other components to data
        _fields2export: list of field names to export to other components from data
    """

    @abc.abstractmethod
    def initialize(self, coupler: "Coupler") -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        raise NotImplementedError

    def finalize(
        self, coupler: "Coupler", output_file_mask: Optional[Path] = None
    ) -> None:
        """Finalize the component by writing its all shared fields (incoming and outgoing)
        to a netCDF file.

        Arguments:
            output_file_mask: optional mask to include in the output filename
        """

        if output_file_mask is None:
            filepath = Path(f"{self.name.lower()}_component_shared_fields.nc")
        else:
            filepath = Path(f"{self.name.lower()}_{output_file_mask}.nc")

        merged_fields = self.merge_incoming_outgoing_fields()
        coupler.append_masks_to_output(self.name, merged_fields)

        write_shared_to_netcdf(merged_fields, self.grid, filepath)

    def check_not_empty_import_export_lists(self) -> None:
        """Check that the component has non-empty and non-overlapping
        import and export fields.
        """

        if not self._fields2import:
            raise ComponentError(
                f"Component '{self.name}' has no fields to import defined."
            )
        if not self._fields2export:
            raise ComponentError(
                f"Component '{self.name}' has no fields to export defined."
            )

        all_fields = set(self._fields2import + self._fields2export)
        if len(all_fields) < len(self._fields2import) + len(self._fields2export):
            raise ComponentError(
                f"Component '{self.name}' has overlapping fields in import/export lists."
            )

    def export_fields(self) -> Shared:
        """
        Prepare and deposit/return the outgoing_fields to be sent/dispatched to another component(s).
        """
        # TODO: export only component related fields
        return self.outgoing_fields

    def import_fields(self, fields: Shared) -> None:
        """
        Import fields received from another component(s) into receptor/incoming_fields.

        Arguments:
            fields: Shared object containing fields to import from another component
        """
        # TODO: import only component related fields

        incoming_fields = fields.field_names
        for name in incoming_fields:
            self.incoming_fields[name] = fields[name]

    def receive_fields(self, time: datetime) -> None:
        """
        Receive interpolated fields from receptor/incoming_fields (from another component(s))
        and store them in data.

        Arguments:
            time: current simulation (coupler's) time
        """

        for fld in self._fields2import:
            try:
                tna = self.incoming_fields[fld]
            except KeyError as exc:
                raise ComponentError(
                    f"Field '{fld}' required by component '{self.name}' not found in incoming fields."
                ) from exc

            if tna is not None and tna.timestamp != time:
                raise ComponentError(
                    f"Receive field '{fld}' timestamp {tna.timestamp} does not match "
                    f"current time {time} in component '{self.name}'."
                )

        self.data.update(self.incoming_fields.fields())

    def send_fields(self, time: datetime, coupler: "Coupler") -> None:
        """
        Prepare fields from data to be deposited to outgoing_fields,
        to be later sent to another component(s).

        Arguments:
            time: current simulation (coupler's) time
            coupler: Coupler instance for possible time interpolation
        """

        for fld in self._fields2export:
            if self._settings.get("apply_time_interpolation", False):
                # for data models with monthly means
                field2send = get_field_at_specific_time(fld, self.data, coupler)
            elif self._settings.get("get_field_time_slice", False):
                # for data models with higher frequency data
                field2send = get_field_time_slice(fld, self.data, time)
            else:
                field2send = self.data[fld]

            self.outgoing_fields[fld] = (field2send, time, self.name)

    def check_valid_exchange_field_names(self) -> None:
        for fld in set(self._fields2import + self._fields2export):
            if fld not in VALID_EXCHANGE_FIELD_NAMES:
                raise ComponentError(
                    f"Field name '{fld}' in component '{self.name}' is not a recognized exchange variable.\n"
                    f"Replace field name '{fld}' with one of the supported names: {VALID_EXCHANGE_FIELD_NAMES}"
                )

    def get(self, field_name: str) -> NDArray:
        """
        Returns the data array of the specified field from either
        incoming_fields or outgoing_fields.

        Arguments:
            field_name (str): name of the field to retrieve
        """

        in_fields = self.incoming_fields.fields()
        out_fields = self.outgoing_fields.fields()

        if field_name in in_fields and field_name in out_fields:
            raise ComponentError(
                f"Field name '{field_name}' found in both incoming and outgoing fields."
            )

        if field_name in in_fields:
            return in_fields[field_name]

        if field_name in out_fields:
            return out_fields[field_name]

        if field_name in self.data:
            return self.data[field_name]

        raise ComponentError(
            f"Field name '{field_name}' not found in incoming, outgoing or internal pool of fields"
        )

    def merge_incoming_outgoing_fields(self) -> Shared:
        """
        Merge incoming_fields and outgoing_fields into a single Shared object for further output.
        """

        output_fields = Shared()

        for name, tna in self.incoming_fields._fields.items():
            output_fields[name] = tna
        for name, tna in self.outgoing_fields._fields.items():
            output_fields[name] = tna

        return output_fields

    def __str__(self) -> str:
        shared_fields_list = []
        shared_fields_string = ""

        if self.incoming_fields or self.outgoing_fields:
            shared_fields_list = list(self.incoming_fields.fields().keys()) + list(
                self.outgoing_fields.fields().keys()
            )
            shared_fields_string = ", ".join(shared_fields_list)

        return (
            f"{self.__class__.__name__}:\n"
            f" ├── Name: {self.name}\n"
            f" ├── Shared fields: {shared_fields_string if len(shared_fields_list) > 0 else 'Not provided'}\n"
            f" └── Grid name: {self.grid.name}\n"
            f"     └── Shape: {self.grid.shape}\n"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, grid={repr(self.grid)},"
            f" incoming_fields={repr(self.incoming_fields)}, outgoing_fields={repr(self.outgoing_fields)})"
        )


class ComponentForcingData:
    def __init__(self) -> None:
        self.DATA_FILES: dict[str, str] = {}

    def _read_forcing(self, variable: str, where: str, flip_y: bool = False) -> NDArray:
        """Read a variable from the specified forcing file.

        Arguments:
            variable (str): variable name to read from a file
            where (str): key to identify which file to read from DATA_FILES
            flip_y (bool): whether to flip the variable along the latitude axis

        Returns:
            (`ndarray`): the requested variable data
        """

        try:
            with h5netcdf.File(self.DATA_FILES[where], "r") as infile:
                var_obj = np.array(infile.variables[variable]).T
                if flip_y:
                    return np.flip(var_obj, axis=1)
                else:
                    return var_obj
        except KeyError as e:
            raise KeyError(
                f"Provided 'where' key '{where}' not found in DATA_FILES"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error reading variable '{variable}' from forcing file '{self.DATA_FILES[where]}'"
            ) from e

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"└── Forcing files: {self.DATA_FILES if self.DATA_FILES else 'No files assigned'}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(DATA_FILES={self.DATA_FILES})"


def write_shared_to_netcdf(
    shared: Shared, grid: RectilinearGrid, filename: Path
) -> None:
    """Write the contents of a Shared object to a netCDF file.
    Arguments:
        shared: Shared object containing fields to write
        grid: Grid object defining the grid
        filename: path to the output netCDF file
    """

    lat = xr.DataArray(grid.latitude, dims=("nlat",), name="latitude")
    lon = xr.DataArray(grid.longitude, dims=("nlon",), name="longitude")

    data_vars = {}
    for name, tna in shared._fields.items():
        data_vars[name] = xr.DataArray(
            data=tna.data,
            dims=("nlat", "nlon"),
            coords={"latitude": lat, "longitude": lon},
            attrs={
                "timestamp": tna.timestamp.isoformat(),
                "component": tna.component_name,
            },
        )

    xr.Dataset(
        data_vars=data_vars,
        coords={"latitude": lat, "longitude": lon},
    ).to_netcdf(filename)


if __name__ == "__main__":
    shared = Shared()
    if not shared.is_empty:
        print("Shared is not empty initially, something is wrong!")

    t_model = datetime(2025, 11, 14, 12, 0, 0)
    shared.temperature = (np.array([[1.0, 2.0], [3.0, 4.0]]), t_model, "ocean")
    shared.humidity = (np.array([[0.5, 0.6], [0.7, 0.8]]), t_model, "atmosphere")
    shared.temperature.data += 10.0

    if shared.is_empty:
        print("Shared is not empty!")

    print(shared)

    temp_array = shared.temperature
    print(temp_array)
    print("Temperature data:\n", temp_array.data)
    print("Temperature timestamp:", temp_array.timestamp)
    print("Temperature component name:", temp_array.component_name)
