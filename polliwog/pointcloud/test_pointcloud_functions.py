import math
import numpy as np
import pytest
from vg.compat import v2 as vg
from ._pointcloud_functions import extent, percentile


def random_points_along_side_of_cylinder(radius=1.0, height=3.0, n_samples=1):
    # Adapted from https://stackoverflow.com/a/9203691
    thetas = np.random.uniform(0, 2 * math.pi, n_samples)
    rs = np.sqrt(np.random.uniform(0, 1, n_samples)) * radius
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    zs = np.random.uniform(0, height, n_samples)
    return np.vstack([xs, ys, zs]).T


def test_percentile():
    points = random_points_along_side_of_cylinder(height=10.0, n_samples=2000)
    result = percentile(points=points, axis=vg.basis.neg_z, percentile=75)

    centroid = np.average(points, axis=0)
    np.testing.assert_almost_equal(result[0], centroid[0])
    np.testing.assert_almost_equal(result[1], centroid[1])
    # 2.5 is the 75th percentile along negative Z.
    np.testing.assert_almost_equal(result[2], 2.5, decimal=1)

    with pytest.raises(ValueError, match="At least one point is needed"):
        percentile(points=np.zeros((0, 3)), axis=vg.basis.neg_z, percentile=75)

    with pytest.raises(ValueError, match="Axis must be non-zero"):
        percentile(points=points, axis=np.array([0, 0, 0]), percentile=75)


def test_extent():
    points = np.array(
        [[0, 0, 0], [10, 10, 0], [10, 0, 0], [0, 11, 0]], dtype=np.float64
    )
    np.testing.assert_almost_equal(extent(points), 14.87, decimal=2)

    distance, i, j = extent(points, ret_indices=True)
    np.testing.assert_almost_equal(distance, 14.87, decimal=2)
    assert i == 2
    assert j == 3

    with pytest.raises(ValueError, match="At least two points are required"):
        extent(np.array([[0, 0, 0]]))
