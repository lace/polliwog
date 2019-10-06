import numpy as np
import pytest
import vg
from .shape import columnize


def test_columnize_2d():
    shape = (-1, 3)

    columnized, is_columnized, transform_result = columnize(vg.basis.x, shape)
    np.testing.assert_array_equal(columnized, np.array([vg.basis.x]))
    assert columnized.shape == (1, 3)
    assert is_columnized is False
    assert transform_result([1.0]) == 1.0

    columnized, is_columnized, transform_result = columnize(
        np.array([vg.basis.x]), shape
    )
    np.testing.assert_array_equal(columnized, np.array([vg.basis.x]))
    assert columnized.shape == (1, 3)
    assert is_columnized is True
    assert transform_result([1.0]) == [1.0]


def test_columnize_3d():
    shape = (-1, 3, 3)

    columnized, is_columnized, transform_result = columnize(
        np.array([vg.basis.x, vg.basis.y, vg.basis.z]), shape
    )
    np.testing.assert_array_equal(
        columnized, np.array([[vg.basis.x, vg.basis.y, vg.basis.z]])
    )
    assert columnized.shape == (1, 3, 3)
    assert is_columnized is False
    assert transform_result([1.0]) == 1.0

    columnized, is_columnized, transform_result = columnize(
        np.array([[vg.basis.x, vg.basis.y, vg.basis.z]]), shape
    )
    np.testing.assert_array_equal(
        columnized, np.array([[vg.basis.x, vg.basis.y, vg.basis.z]])
    )
    assert columnized.shape == (1, 3, 3)
    assert is_columnized is True
    assert transform_result([1.0]) == [1.0]


def test_columnize_invalid_shape():
    with pytest.raises(ValueError, match="shape should be a tuple"):
        columnize(vg.basis.x, "this is not a shape")
