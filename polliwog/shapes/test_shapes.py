import numpy as np
import pytest
from ._shapes import cube, rectangular_prism, triangular_prism


def test_rectangular_prism():
    origin = np.array([3.0, 4.0, 5.0])
    size = np.array([2.0, 10.0, 20.0])

    expected_vertices = np.array(
        [
            [3.0, 4.0, 5.0],
            [5.0, 4.0, 5.0],
            [5.0, 4.0, 25.0],
            [3.0, 4.0, 25.0],
            [3.0, 14.0, 5.0],
            [5.0, 14.0, 5.0],
            [5.0, 14.0, 25.0],
            [3.0, 14.0, 25.0],
        ]
    )
    expected_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [7, 6, 5],
            [7, 5, 4],
            [4, 5, 1],
            [4, 1, 0],
            [5, 6, 2],
            [5, 2, 1],
            [6, 7, 3],
            [6, 3, 2],
            [3, 7, 4],
            [3, 4, 0],
        ]
    )

    vertices, faces = rectangular_prism(
        origin=origin, size=size, ret_unique_vertices_and_faces=True
    )
    np.testing.assert_array_equal(faces, expected_faces)
    np.testing.assert_array_equal(vertices, expected_vertices)

    flattened_vertices = rectangular_prism(
        origin=origin, size=size, ret_unique_vertices_and_faces=False
    )
    np.testing.assert_array_equal(flattened_vertices, expected_vertices[expected_faces])


def test_cube():
    origin = np.array([3.0, 4.0, 5.0])
    size = 2.0

    flattened_vertices = cube(origin=origin, size=size)

    expected_first_triangle = np.array(
        [[3.0, 4.0, 5.0], [5.0, 4.0, 5.0], [5.0, 4.0, 7.0]]
    )
    np.testing.assert_array_equal(flattened_vertices[0], expected_first_triangle)

    with pytest.raises(ValueError, match="`size` should be a number"):
        cube(origin=origin, size="not a number")


def test_triangular_prism():
    p1 = np.array([3.0, 0.0, 0.0])
    p2 = np.array([0.0, 3.0, 0.0])
    p3 = np.array([0.0, 0.0, 3.0])

    flattened_vertices = triangular_prism(p1, p2, p3, 1.0)

    expected_first_triangle = np.array([p1, p2, p3])
    np.testing.assert_array_equal(flattened_vertices[0], expected_first_triangle)

    with pytest.raises(ValueError, match="`height` should be a number"):
        triangular_prism(p1, p2, p3, "not-a-number")
