import math
import numpy as np
import vg
from .equations import plane_equation_from_points, plane_normal_from_points


def assert_plane_equation_satisfies_points(plane_equation, points):
    a, b, c, d = plane_equation
    plane_equation_test = [a * x + b * y + c * z + d for x, y, z in points]
    assert np.any(plane_equation_test) == False


def test_plane_equation_from_points():
    points = np.array([[1, 1, 1], [-1, 1, 0], [2, 0, 3]])
    equation = plane_equation_from_points(points)
    assert_plane_equation_satisfies_points(equation, points)


def test_plane_equation_from_points_is_in_expected_orientation():
    equation = plane_equation_from_points(
        np.array([vg.basis.x, vg.basis.y, vg.basis.neg_x])
    )
    normal = equation[:3]
    np.testing.assert_array_equal(normal, vg.basis.z)


def test_plane_equation_from_points_stacked():
    points = np.array(
        [[[1, 1, 1], [-1, 1, 0], [2, 0, 3]], [vg.basis.x, vg.basis.y, vg.basis.neg_x]]
    )
    equations = plane_equation_from_points(points)
    for these_points, this_equation in zip(points, equations):
        assert_plane_equation_satisfies_points(this_equation, these_points)


def test_plane_normal_from_points():
    np.testing.assert_array_almost_equal(
        plane_normal_from_points(np.array([vg.basis.x, vg.basis.y, vg.basis.neg_x])),
        vg.basis.z,
    )


def test_plane_normal_from_points_stacked():
    np.testing.assert_array_almost_equal(
        plane_normal_from_points(
            np.array(
                [
                    [vg.basis.x, vg.basis.y, vg.basis.z],
                    [vg.basis.x, vg.basis.y, vg.basis.neg_x],
                ]
            )
        ),
        np.array([np.repeat(math.sqrt(1.0 / 3.0), 3), vg.basis.z]),
    )


def normal_and_offset_from_plane_equations():
    points = np.array(
        [[[1, 1, 1], [-1, 1, 0], [2, 0, 3]], [vg.basis.x, vg.basis.y, vg.basis.neg_x]]
    )
    equations = plane_equation_from_points(points)
    normals, offsets = equations
    np.testing.assert_array_almost_equal(
        normals, np.array([np.repeat(math.sqrt(1.0 / 3.0), 3), vg.basis.z])
    )
