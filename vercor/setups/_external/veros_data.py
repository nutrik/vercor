"""Copy prescribed monthly ocean surfaces and geometry from Veros global 4°.

Native geometry has physical (x, y[, z]) axes, with two halo cells on each
horizontal boundary. Surface forcing is transposed once to VerCOR's
(month, latitude, longitude) layout. Veros depth is positive downward; Veris
R_low is sea-floor elevation, so ht changes sign once here. Spacings (m) and
spherical cell areas (m²) are already native metrics and are not rescaled.
No prognostic ocean step or native owner is retained by the snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


def _copy_arrays(values: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
    """Own read-only host arrays without selecting runtime model precision."""
    result: dict[str, RuntimeArray] = {}
    for name, value in values.items():
        array = np.array(value, copy=True)
        array.flags.writeable = False
        result[name] = array
    return MappingProxyType(result)


@dataclass(frozen=True)
class VerosDataSnapshot:
    """Independent geometry and monthly surface data on one shared ocean grid."""

    grid: RectilinearGrid
    geometry: Mapping[str, RuntimeArray]
    fields: Mapping[str, RuntimeArray]

    def __post_init__(self) -> None:
        """Copy all geometry and forcing arrays into read-only owned mappings."""
        object.__setattr__(self, "geometry", _copy_arrays(self.geometry))
        object.__setattr__(self, "fields", _copy_arrays(self.fields))


def snapshot_veros_state(state: Any) -> VerosDataSnapshot:
    """Extract physical native interiors without retaining the mutable state."""
    vs = state.variables
    geometry = {
        name: np.asarray(getattr(vs, name))[2:-2, 2:-2]
        for name in (
            "maskT",
            "maskU",
            "maskV",
            "ht",
            "coriolis_t",
            "area_t",
            "area_u",
            "area_v",
        )
    }
    geometry["ht"] = -geometry["ht"]
    geometry.update(
        {
            name: np.asarray(getattr(vs, name))[2:-2]
            for name in ("dxt", "dxu", "dyt", "dyu")
        }
    )
    grid = RectilinearGrid(
        "veros-global-4deg",
        longitude=np.asarray(vs.xt)[2:-2],
        latitude=np.asarray(vs.yt)[2:-2],
        longitude_edges=np.asarray(vs.xu)[1:-2],
        latitude_edges=np.asarray(vs.yu)[1:-2],
        binary_mask=geometry["maskT"][:, :, -1].T.astype(float),
    )
    sst = np.asarray(vs.sst_clim)[2:-2, 2:-2].transpose(2, 1, 0) + 273.15
    surface_fields = {
        "sea_surface_temperature": sst,
        "sea_surface_salinity": np.asarray(vs.sss_clim)[2:-2, 2:-2].transpose(2, 1, 0),
        # Prescribed quiescent ocean and zero height anomaly, not native initial
        # velocity allocations presented as an equilibrated ocean solution.
        "ocean_u_velocity": np.zeros_like(sst),
        "ocean_v_velocity": np.zeros_like(sst),
        "sea_surface_height": np.zeros_like(sst),
    }
    return VerosDataSnapshot(grid=grid, geometry=geometry, fields=surface_fields)


def _initialize_veros() -> Any:
    """Initialize native global geometry once in a host-compatible process."""
    try:
        import veros  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Prescribed Veros data requires `pip install 'vercor[veros]'`."
        ) from error
    from vercor.setups._external.veros_runtime_settings import configure_veros_runtime

    configure_veros_runtime("host")
    from vercor.setups._external.veros_setup import CustomGlobalFourDegree

    model = CustomGlobalFourDegree()
    model.setup()
    return model.state


def load_veros_data_snapshot() -> VerosDataSnapshot:
    """Load real global-four-degree geometry/climatology without ocean stepping."""
    return snapshot_veros_state(_initialize_veros())
