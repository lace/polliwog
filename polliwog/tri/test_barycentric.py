import math
import numpy as np
from .barycentric import compute_barycentric_coordinates


def test_barycentric():
    triangle = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, math.sqrt(2), 0.0]])

    points_of_interest = np.array(
        [
            [0.0, math.sqrt(2) / 3.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, math.sqrt(2), 0.0],
        ]
    )
    expected = np.array(
        [np.full((3,), 1.0 / 3.0), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # Perform all of our queries on the same triangle.
    tiled_triangle = np.tile(triangle.reshape(-1), len(points_of_interest)).reshape(
        -1, 3, 3
    )

    np.testing.assert_array_almost_equal(
        compute_barycentric_coordinates(tiled_triangle, points_of_interest), expected
    )
