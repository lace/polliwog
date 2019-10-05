import math
import numpy as np
import vg
from . import equations


def assert_plane_equation_satisfies_points(plane_equation, points):
    a, b, c, d = plane_equation
    plane_equation_test = [a * x + b * y + c * z + d for x, y, z in points]
    assert np.any(plane_equation_test) == False


def test_plane_equation_from_points():
    points = np.array([[1, 1, 1], [-1, 1, 0], [2, 0, 3]])
    equation = equations.plane_equation_from_points(points)
    assert_plane_equation_satisfies_points(equation, points)


def test_plane_equation_from_points_is_in_expected_orientation():
    # Set up.
    points = np.array([vg.basis.x, vg.basis.y, vg.basis.neg_x])

    # Act.
    equation = equations.plane_equation_from_points(points)

    # Confidence check.
    assert_plane_equation_satisfies_points(equation, points)

    # Assert.
    normal = equation[:3]
    np.testing.assert_array_equal(normal, vg.basis.z)
