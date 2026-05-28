from __future__ import annotations

import pytest

from vercor.field_layout import (
    canonical_grid_field_shape,
    canonical_grid_field_shape_error,
    validate_canonical_grid_field_shape,
)


def test_canonical_grid_field_shape_error_is_shared() -> None:
    error = canonical_grid_field_shape_error(
        field_name="temperature",
        shape=(3,),
        grid_shape=(2, 2),
        owner_description="Runtime required data field",
        owner_name="ATM",
    )

    assert "Runtime required data field 'temperature' for component 'ATM'" in error
    assert "(nLat, nLon)" in error
    assert "trailing grid shape (2, 2)" in error


def test_validate_canonical_grid_field_shape_raises_consistent_error() -> None:
    with pytest.raises(ValueError, match="Runtime required data field 'humidity'"):
        validate_canonical_grid_field_shape(
            field_name="humidity",
            value=[1.0, 2.0],
            grid_shape=(2, 2),
            owner_description="Runtime required data field",
            owner_name="ATM",
        )


def test_canonical_grid_field_shape_normalizes_array_shape() -> None:
    assert canonical_grid_field_shape([[1.0, 2.0], [3.0, 4.0]]) == (2, 2)
