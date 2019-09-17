import numpy as np
from .shapes import create_rectangular_prism, create_cube, create_horizontal_plane


def test_create_rectangular_prism():
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

    vertices, faces = create_rectangular_prism(
        origin=origin, size=size, ret_unique_vertices_and_faces=True
    )
    np.testing.assert_array_equal(faces, expected_faces)
    np.testing.assert_array_equal(vertices, expected_vertices)

    flattened_vertices = create_rectangular_prism(
        origin=origin, size=size, ret_unique_vertices_and_faces=False
    )
    np.testing.assert_array_equal(flattened_vertices, expected_vertices[expected_faces])


def test_create_cube():
    origin = np.array([3.0, 4.0, 5.0])
    size = 2.0

    flattened_vertices = create_cube(origin=origin, size=size)

    expected_first_triangle = np.array(
        [[3.0, 4.0, 5.0], [5.0, 4.0, 5.0], [5.0, 4.0, 7.0]]
    )
    np.testing.assert_array_equal(flattened_vertices[0], expected_first_triangle)


def test_create_horizontal_plane():
    expected_vertices = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    expected_faces = np.array([[0, 1, 2], [3, 1, 0]])

    vertices, faces = create_horizontal_plane(ret_unique_vertices_and_faces=True)
    np.testing.assert_array_equal(faces, expected_faces)
    np.testing.assert_array_equal(vertices, expected_vertices)
