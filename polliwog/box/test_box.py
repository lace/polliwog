import numpy as np
import pytest
from ._box_object import Box


def create_box():
    return Box(np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 5.0]))


def test_ranges():
    np.testing.assert_array_equal(
        create_box().ranges, np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 8.0]])
    )


def test_mins_mids_maxes():
    box = create_box()
    assert box.min_x == 1.0
    assert box.min_y == 2.0
    assert box.min_z == 3.0
    assert box.mid_x == 2.5
    assert box.mid_y == 2.5
    assert box.mid_z == 5.5
    assert box.max_x == 4.0
    assert box.max_y == 3.0
    assert box.max_z == 8.0


def test_dims():
    box = create_box()
    assert box.width == 3.0
    assert box.height == 1.0
    assert box.depth == 5.0


def test_center_point():
    np.testing.assert_array_equal(create_box().center_point, np.array([2.5, 2.5, 5.5]))


def test_floor_point():
    np.testing.assert_array_equal(create_box().floor_point, np.array([2.5, 2.0, 5.5]))


def test_volume():
    assert create_box().volume == 15.0


def test_surface_area():
    assert create_box().surface_area == 46.0


def test_box_from_points():
    box = Box.from_points(
        np.array([[1.0, 2.0, 3.0], [-3.0, 4.0, 5.0], [4.0, 0.0, -6.0]])
    )
    np.testing.assert_array_equal(box.origin, np.array([-3.0, 0.0, -6.0]))
    np.testing.assert_array_equal(box.size, np.array([7.0, 4.0, 11.0]))

    with pytest.raises(ValueError, match="Need at least 1 point"):
        Box.from_points(np.zeros((0, 3)))


def test_invalid_shape():
    with pytest.raises(ValueError):
        Box(np.array([1.0, 2.0, 3.0]), np.array([-1.0, 1.0, 1.0]))


def test_v():
    box = create_box()
    assert box.v.shape == (8, 3)
    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 2.0, 3.0],
            [1.0, 3.0, 3.0],
            [1.0, 2.0, 8.0],
            [4.0, 3.0, 3.0],
            [1.0, 3.0, 8.0],
            [4.0, 2.0, 8.0],
            [4.0, 3.0, 8.0],
        ]
    )
    np.testing.assert_array_equal(box.v, expected)


def test_contains():
    box = create_box()
    assert box.contains(np.array([1.1, 2.2, 3.3]))
    assert box.contains(np.array([1.0, 2.0, 3.0]))
    assert box.contains(np.array([0.99, 2.0, 3.0])) == False  # noqa: E712
    assert box.contains(np.array([0.99, 2.0, 3.0]), atol=1e-2) == True  # noqa: E712
