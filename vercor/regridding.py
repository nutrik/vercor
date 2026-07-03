"""Public regridding facade for VerCOR grid-to-grid transfers."""

from vercor._deprecation import deprecated_getattr
from vercor.regridders import (
    BilinearRectilinearRegridder,
    ConservativeRectilinearRegridder,
    Regridder,
    bilinear,
    conservative,
)

__all__ = [
    "BilinearRectilinearRegridder",
    "ConservativeRectilinearRegridder",
    "Regridder",
    "bilinear",
    "conservative",
]


__getattr__ = deprecated_getattr(
    __name__,
    {
        "BilinearRegridder": (
            "vercor.regridding.BilinearRectilinearRegridder",
            BilinearRectilinearRegridder,
        ),
        "ConservativeRegridder": (
            "vercor.regridding.ConservativeRectilinearRegridder",
            ConservativeRectilinearRegridder,
        ),
    },
    remove_in="0.2.0",
)
