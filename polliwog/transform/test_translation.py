import numpy as np
from .translation import translation


def create_points():
    # test data: square of side 1 with lower left corner at (1, 1)
    return np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0]])


def test_translates_and_returns_translation_factor_with_translation_factor():
    points = create_points()
    translation_factor = np.array([-1.0, 2.0])

    expected_image = np.array([[0.0, 3.0], [0.0, 4.0], [1.0, 4.0], [1.0, 3.0]])

    image, returned_translation_factor = translation(points, translation_factor)

    np.testing.assert_array_equal(image, expected_image)
    np.testing.assert_array_equal(returned_translation_factor, translation_factor)


def test_translates_with_center_of_mass_and_returns_translation_factor_with_no_translation_factor():
    points = create_points()
    expected_image = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
    expected_translation_factor = -np.mean(points, axis=0)

    image, translation_factor = translation(points)

    np.testing.assert_array_equal(image, expected_image)
    np.testing.assert_array_equal(translation_factor, expected_translation_factor)
