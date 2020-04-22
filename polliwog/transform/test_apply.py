import numpy as np
from ._apply import apply_transform


def test_apply_transform():
    scale_factor = np.array([3.0, 0.5, 2.0])
    transform = np.array(
        [
            [scale_factor[0], 0, 0, 0],
            [0, scale_factor[1], 0, 0],
            [0, 0, scale_factor[2], 0],
            [0, 0, 0, 1],
        ]
    )

    points = np.array([[1.0, 2.0, 3.0], [5.0, 0.0, 1.0]])
    expected_points = np.array([[3.0, 1.0, 6.0], [15.0, 0.0, 2.0]])

    transformer = apply_transform(transform)
    np.testing.assert_array_equal(transformer(points), expected_points)
    np.testing.assert_array_equal(transformer(points[1]), expected_points[1])

    expected_points_discarding_z = np.array([[3.0, 1.0], [15.0, 0.0]])
    np.testing.assert_array_equal(
        transformer(points, discard_z_coord=True), expected_points_discarding_z,
    )
    np.testing.assert_array_equal(
        transformer(points[1], discard_z_coord=True), expected_points_discarding_z[1],
    )

def test_apply_transform_for_vectors():
    translation = np.array([3.0, 0.5, 2.0])
    transform = np.array(
        [
            [1, 0, 0, translation[0]],
            [0, 1, 0, translation[1]],
            [0, 0, 1, translation[2]],
            [0, 0, 0, 1],
        ]
    )

    points = np.array([[1.0, 2.0, 3.0], [5.0, 0.0, 1.0]])
    expected_points = np.array([[4.0, 2.5, 5.0], [8.0, 0.5, 3.0]])

    # Confidence check.
    transformer = apply_transform(transform)
    np.testing.assert_array_equal(transformer(points), expected_points)

    np.testing.assert_array_equal(transformer(points, treat_input_as_vector=True), points)
