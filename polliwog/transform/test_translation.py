import unittest
import numpy as np
from blmath.geometry.transform.translation import translation

class TranslationsTest(unittest.TestCase):

    def setUp(self):
        self.translation = translation

        # test data: square of side 1 with lower left corner at (1, 1)
        self.points = np.array([
            [1., 1.],
            [1., 2.],
            [2., 2.],
            [2., 1.],
        ])

    def test_translates_and_returns_translation_factor_with_translation_factor(self):
        points = self.points
        translation_factor = np.array([-1., 2.])

        expected_image = np.array([
            [0., 3.],
            [0., 4.],
            [1., 4.],
            [1., 3.],
        ])

        image, returned_translation_factor = translation(points, translation_factor)

        np.testing.assert_array_equal(image, expected_image)
        np.testing.assert_array_equal(returned_translation_factor, translation_factor)

    def test_translates_with_center_of_mass_and_returns_translation_factor_with_no_translation_factor(self):
        points = self.points
        expected_image = np.array([
            [-.5, -.5],
            [-.5, .5],
            [.5, .5],
            [.5, -.5],
        ])
        expected_translation_factor = -np.mean(points, axis=0)

        image, translation_factor = translation(points)

        np.testing.assert_array_equal(image, expected_image,)
        np.testing.assert_array_equal(translation_factor, expected_translation_factor)
