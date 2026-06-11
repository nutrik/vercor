from __future__ import annotations

from datetime import datetime

import numpy as np

from vercor.output.datasets import time_coordinate_variable, used_dimension_names
from vercor.output.variables import OutputVariable


def test_time_coordinate_variable_preserves_calendar_attrs() -> None:
    variable = time_coordinate_variable(datetime(2000, 1, 2, 3, 4, 5))

    assert variable.dims == ("time",)
    assert variable.attrs["calendar"] == "proleptic_gregorian"
    assert variable.attrs["isoformat"] == "2000-01-02T03:04:05"
    assert np.asarray(variable.values).shape == (1,)


def test_used_dimension_names_preserves_first_use_order_and_excludes_time() -> None:
    variables = {
        "temp": OutputVariable(("time", "zt", "yt", "xt"), np.ones((1, 2, 3, 4))),
        "psi": OutputVariable(("time", "yu", "xu"), np.ones((1, 5, 6))),
        "salt": OutputVariable(("time", "yt", "xt"), np.ones((1, 3, 4))),
    }

    assert used_dimension_names(variables) == ("zt", "yt", "xt", "yu", "xu")
