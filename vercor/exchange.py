from dataclasses import dataclass, field
from typing import Callable, Union

from vercor.grid import Grid
from vercor.regridders.bilinear import BilinearRectilinearRegridder
from vercor.regridders.conservative import ConservativeRectilinearRegridder


VALID_EXCHANGE_FIELD_NAMES: list[str] = [
    "specific_humidity",
    "temperature",
    "temperature_2m",
    "potential_temperature",
    "sea_surface_temperature",
    "land_surface_temperature",
    "model_level_height",
    "u_velocity",
    "v_velocity",
    "u_velocity_10m",
    "v_velocity_10m",
    "surface_pressure",
    "pressure",
    "density",
    "ice_fraction",
    "soil_moisture",
    "sensible_heat_flux",
    "latent_heat_flux",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
]


@dataclass
class Exchange:
    source: str
    destination: str
    name: str = field(init=False)
    field_names: list[Union[str, tuple[str, str]]]
    regridder_factory: Callable[
        ..., BilinearRectilinearRegridder | ConservativeRectilinearRegridder
    ]
    interpolation_type: str = field(init=False)
    """
    Exchange definition between two components

        source, destination: component names
        name: exchange name (automatically set to "SOURCE2DESTINATION")
        field_names: list of scalar field names and
                tuples of vectors (u-component, v-component)
        regridder_factory: callable that returns a Regridder instance
        interpolation_type: type of interpolation used (automatically set from regridder_factory)
    """

    def __post_init__(self) -> None:
        self.name = (
            f"{self.source} --({self.regridder_factory.__name__})--> {self.destination}"
        )
        self.interpolation_type = self.regridder_factory.__name__

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

    def create(
        self,
        source_grid: Grid,
        destination_grid: Grid,
    ) -> BilinearRectilinearRegridder | ConservativeRectilinearRegridder:
        """
        Create and return a Regridder instance using the provided factory.

        Arguments:
            source_grid: Grid of the source component
            destination_grid: Grid of the destination component

        Returns:
            Regridder instance created by the factory
        """
        return self.regridder_factory(source_grid, destination_grid)
