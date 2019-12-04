import math
import numpy as np
import vg
from .functions import (
    barycentric_coordinates_of_points,
    surface_normals,
    tri_contains_coplanar_point,
)


def test_surface_normals_from_points_single():
    points = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

    np.testing.assert_allclose(
        surface_normals(points), np.array([3 ** -0.5, 3 ** -0.5, 3 ** -0.5])
    )

    np.testing.assert_allclose(
        surface_normals(points, normalize=False), np.array([9.0, 9.0, 9.0])
    )


def test_surface_normals_from_points_vectorized():
    from ..shapes import create_triangular_prism

    p1 = np.array([3.0, 0.0, 0.0])
    p2 = np.array([0.0, 3.0, 0.0])
    p3 = np.array([0.0, 0.0, 3.0])
    vertices = create_triangular_prism(p1, p2, p3, 1.0)

    expected_normals = vg.normalize(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -2.0],
                [1.0, 1.0, -2.0],
                [-2.0, 1.0, 1.0],
                [-2.0, 1.0, 1.0],
                [1.0, -2.0, 1.0],
                [1.0, -2.0, 1.0],
                [-1.0, -1.0, -1.0],
            ]
        )
    )

    np.testing.assert_allclose(surface_normals(vertices), expected_normals)


def test_tri_contains_coplanar_point():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([4.0, 0.1, 0.0])
    c = np.array([3.0, 3.1, 0.0])

    # Not sure why, but `is True` does not work.
    assert tri_contains_coplanar_point(a, b, c, a) == True  # noqa: E712
    assert tri_contains_coplanar_point(a, b, c, b) == True  # noqa: E712
    assert tri_contains_coplanar_point(a, b, c, c) == True  # noqa: E712
    assert (
        tri_contains_coplanar_point(a, b, c, np.array([2.0, 1.0, 0.0])) == True
    )  # noqa: E712

    # Unexpected, as it's not in the plane, though if projected to the plane,
    # it is in the triangle.
    assert (
        tri_contains_coplanar_point(a, b, c, np.array([0.0, 0.0, 1.0])) == True
    )  # noqa: E712

    assert (
        tri_contains_coplanar_point(a, b, c, np.array([2.0, 0.0, 0.0])) == False
    )  # noqa: E712
    assert (
        tri_contains_coplanar_point(a, b, c, np.array([2.0, 5.0, 0.0])) == False
    )  # noqa: E712

    assert (
        tri_contains_coplanar_point(
            np.array([0.06710189, 1.69908346, 0.06590126]),
            np.array([0.05648619, 1.70207, 0.07402092]),
            np.array([0.05969098, 1.69641423, 0.07268801]),
            np.array([0.07534771, 1.6869296, 0.06190757]),
        )
        == False
    )  # noqa: E712


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
        barycentric_coordinates_of_points(tiled_triangle, points_of_interest), expected
    )
