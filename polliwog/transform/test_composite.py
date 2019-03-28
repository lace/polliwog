import unittest
import numpy as np
from blmath.geometry.transform.composite import CompositeTransform

def create_cube_verts(origin, size):
    # Create a cube. Since CompositeTransform just works on verticies,
    # we don't need a full lace.mesh object.
    origin = np.asarray(origin)
    size = np.repeat(size, 3)
    lower_base_plane = np.array([
        # Lower base plane
        origin,
        origin + np.array([size[0], 0, 0]),
        origin + np.array([size[0], 0, size[2]]),
        origin + np.array([0, 0, size[2]]),
    ])
    upper_base_plane = lower_base_plane + np.array([0, size[1], 0])
    return np.vstack([lower_base_plane, upper_base_plane])

class TestCompositeTransform(unittest.TestCase):

    def setUp(self):
        self.cube_v = create_cube_verts([1., 0., 0.], 4.)

    def test_translate(self):
        transform = CompositeTransform()
        transform.translate(np.array([8., 6., 7.]))

        # Sanity checking.
        np.testing.assert_array_equal(self.cube_v[0], [1., 0., 0.])
        np.testing.assert_array_equal(self.cube_v[6], [5., 4., 4.])

        self.cube_v = transform(self.cube_v)

        np.testing.assert_array_equal(self.cube_v[0], [9., 6., 7.])
        np.testing.assert_array_equal(self.cube_v[6], [13., 10., 11.])

    def test_translate_by_list(self):
        transform = CompositeTransform()
        transform.translate([8., 6., 7.])

        # Sanity checking.
        np.testing.assert_array_equal(self.cube_v[0], [1., 0., 0.])
        np.testing.assert_array_equal(self.cube_v[6], [5., 4., 4.])

        self.cube_v = transform(self.cube_v)

        np.testing.assert_array_equal(self.cube_v[0], [9., 6., 7.])
        np.testing.assert_array_equal(self.cube_v[6], [13., 10., 11.])

    def test_scale(self):
        transform = CompositeTransform()
        transform.scale(10.)

        # Sanity checking.
        np.testing.assert_array_equal(self.cube_v[0], [1., 0., 0.])
        np.testing.assert_array_equal(self.cube_v[6], [5., 4., 4.])

        self.cube_v = transform(self.cube_v)

        np.testing.assert_array_equal(self.cube_v[0], [10., 0., 0.])
        np.testing.assert_array_equal(self.cube_v[6], [50., 40., 40.])

    def test_translate_then_scale(self):
        transform = CompositeTransform()
        transform.translate(np.array([8., 6., 7.]))
        transform.scale(10.)

        # Sanity checking.
        np.testing.assert_array_equal(self.cube_v[0], [1., 0., 0.])
        np.testing.assert_array_equal(self.cube_v[6], [5., 4., 4.])

        self.cube_v = transform(self.cube_v)

        np.testing.assert_array_equal(self.cube_v[0], [90., 60., 70.])
        np.testing.assert_array_equal(self.cube_v[6], [130., 100., 110.])

    def test_scale_then_translate(self):
        transform = CompositeTransform()
        transform.scale(10.)
        transform.translate(np.array([8., 6., 7.]))

        # Sanity checking.
        np.testing.assert_array_equal(self.cube_v[0], [1., 0., 0.])
        np.testing.assert_array_equal(self.cube_v[6], [5., 4., 4.])

        self.cube_v = transform(self.cube_v)

        np.testing.assert_array_equal(self.cube_v[0], [18., 6., 7.])
        np.testing.assert_array_equal(self.cube_v[6], [58., 46., 47.])

    def test_rotate_then_translate(self):
        transform = CompositeTransform()
        transform.rotate(np.array([1., 2., 3.]))
        transform.translate(np.array([3., 2., 1.]))

        v = np.array([1., 0., 0.]).reshape(-1, 3)

        # Forward.
        np.testing.assert_allclose(
            np.array([2.30507944, 1.80799303, 1.69297817]).reshape(-1, 3),
            transform(v))
        # Reverse.
        np.testing.assert_allclose(
            np.array([1.08087689, -1.45082159, -2.3930779]).reshape(-1, 3),
            transform(v, reverse=True))

    def test_reorient(self):
        # TODO We should also test a non-axis-aligned up and look.

        transform = CompositeTransform()
        transform.reorient(up=np.array([0., 1., 0.]), look=np.array([-1., 0., 0.]))

        # Sanity checking.
        np.testing.assert_array_equal(self.cube_v[0], [1., 0., 0.])
        np.testing.assert_array_equal(self.cube_v[6], [5., 4., 4.])

        self.cube_v = transform(self.cube_v)

        np.testing.assert_array_equal(self.cube_v[0], [0., 0., -1.])
        np.testing.assert_array_equal(self.cube_v[6], [4, 4., -5.])

    def test_rotate(self):
        ways_to_rotate_around_y_a_quarter_turn = [
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            np.array([0, np.pi/2, 0]),
            np.array([[0, np.pi/2, 0]]),
            np.array([[0], [np.pi/2], [0]]),
            [0, np.pi/2, 0],
        ]
        for rot in ways_to_rotate_around_y_a_quarter_turn:
            transform = CompositeTransform()
            transform.rotate(rot)
            cube_v = self.cube_v.copy()

            # Sanity checking.
            np.testing.assert_array_equal(cube_v[0], [1., 0., 0.])
            np.testing.assert_array_equal(cube_v[6], [5., 4., 4.])

            cube_v = transform(cube_v)

            np.testing.assert_array_almost_equal(cube_v[0], [0., 0., -1.])
            np.testing.assert_array_almost_equal(cube_v[6], [4, 4., -5.])

    def test_reverse_transforms(self):
        transforms = [CompositeTransform() for _ in range(5)]

        transforms[1].translate(np.array([8., 6., 7.]))

        transforms[2].scale(10.)

        transforms[3].translate(np.array([8., 6., 7.]))
        transforms[3].scale(10.)

        transforms[4].scale(10.)
        transforms[4].translate(np.array([8., 6., 7.]))

        for transform in transforms:
            # Sanity checking.
            np.testing.assert_array_equal(self.cube_v[0], [1., 0., 0.])
            np.testing.assert_array_equal(self.cube_v[6], [5., 4., 4.])

            transformed = transform(self.cube_v)

            untransformed_v = transform(transformed, reverse=True)

            np.testing.assert_array_almost_equal(untransformed_v[0], [1., 0., 0.])
            np.testing.assert_array_almost_equal(untransformed_v[6], [5., 4., 4.])

    def test_forward_reverse_equivalence(self):
        transform = CompositeTransform()
        transform.rotate(np.array([1., 2., 3.]))
        transform.translate(np.array([3., 2., 1.]))
        transform.scale(10.)
        transform.rotate(np.array([7., 13., 5.]))

        forward = transform.matrix_for()
        reverse = transform.matrix_for(reverse=True)
        np.testing.assert_allclose(reverse, np.linalg.inv(forward))

        forward = transform.matrix_for(from_range=(0, 2))
        reverse = transform.matrix_for(from_range=(0, 2), reverse=True)
        np.testing.assert_allclose(reverse, np.linalg.inv(forward))
