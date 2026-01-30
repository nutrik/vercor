from typing import TYPE_CHECKING, Union


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
    )


type OceanType = Union[Ocean, ERA5Ocean, ERAInterimOcean, VerosGCM]
type LandType = Union[Land, ERA5Land, JCMLand]
type AtmosphereType = Union[Atmosphere, ERA5Atmosphere, JAXGCM]
type AllComponentsType = Union[OceanType, LandType, AtmosphereType, SeaIce]
