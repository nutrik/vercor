from vercor._deprecation import deprecated_getattr
from vercor.regridders.base import Regridder
from vercor.regridders.bilinear import (
    BilinearRectilinearRegridder,
    bilinear,
)
from vercor.regridders.conservative import (
    ConservativeRectilinearRegridder,
    conservative,
)

__all__ = [
    "Regridder",
    "BilinearRectilinearRegridder",
    "ConservativeRectilinearRegridder",
    "bilinear",
    "conservative",
]


__getattr__ = deprecated_getattr(
    __name__,
    {
        "BilinearRegridder": (
            "vercor.regridders.BilinearRectilinearRegridder",
            BilinearRectilinearRegridder,
        ),
        "ConservativeRegridder": (
            "vercor.regridders.ConservativeRectilinearRegridder",
            ConservativeRectilinearRegridder,
        ),
    },
    remove_in="0.2.0",
)
