from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    from vercor.components import (
        Atmosphere,
        Land,
        Ocean,
        SeaIce,
        ERA5Atmosphere,
        ERA5Land,
        JCMLand,
        ERA5Ocean,
        ERAInterimOcean,
        JAXGCM,
        VerosGCM,
        CAMulatorGCM,
    )

OceanType: TypeAlias = Union["Ocean", "ERA5Ocean", "ERAInterimOcean", "VerosGCM"]
LandType: TypeAlias = Union["Land", "ERA5Land", "JCMLand"]
AtmosphereType: TypeAlias = Union["Atmosphere", "ERA5Atmosphere", "JAXGCM", "CAMulatorGCM"]
AllComponentsType: TypeAlias = Union[OceanType, LandType, AtmosphereType, "SeaIce"]

