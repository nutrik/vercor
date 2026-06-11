from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vercor.output.netcdf import write_netcdf_dataset
from vercor.output.variables import OutputVariable


def test_write_netcdf_dataset_rejects_conflicting_dimension_sizes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "conflicting-dimensions.nc"

    with pytest.raises(ValueError, match="dimension 'x'.*existing size 2.*new size 3"):
        write_netcdf_dataset(
            output=str(output),
            coordinate_variables={
                "x": OutputVariable(("x",), np.asarray([0.0, 1.0])),
            },
            data_variables={
                "bad": OutputVariable(("x",), np.asarray([0.0, 1.0, 2.0])),
            },
        )
