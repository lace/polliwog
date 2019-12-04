import numpy as np

__all__ = [
    "create_rectangular_prism",
    "create_cube",
    "create_triangular_prism",
    "create_rectangle",
]


def _maybe_flatten(vertices, faces, ret_unique_vertices_and_faces):
    if ret_unique_vertices_and_faces:
        return vertices, faces
    else:
        return vertices[faces]


def create_rectangular_prism(origin, size, ret_unique_vertices_and_faces=False):
    """
    Tesselate an axis-aligned rectangular prism. One vertex is `origin`. The
    diametrically opposite vertex is `origin + size`.

    Args:
        origin (np.ndarray): A 3D point vector containing the point on the
            prism with the minimum x, y, and z coords.
        size (np.ndarray): A 3D vector specifying the prism's length, width,
            and height, which should be positive.
        ret_unique_vertices_and_faces (bool): When `True` return a vertex
            array containing the unique vertices and an array of faces (i.e.
            vertex indices). When `False`, return a flattened array of
            triangle coordinates.

    Returns:
        object:

        - With `ret_unique_vertices_and_faces=True`: a tuple containing
          an `8x3` array of vertices and a `12x3` array of triangle faces.
        - With `ret_unique_vertices_and_faces=False`: a `12x3x3` matrix of
          flattened triangle coordinates.
    """
    from ..tri.quad_faces import quads_to_tris

    lower_base_plane = np.array(
        [
            # Lower base plane
            origin,
            origin + np.array([size[0], 0, 0]),
            origin + np.array([size[0], 0, size[2]]),
            origin + np.array([0, 0, size[2]]),
        ]
    )
    upper_base_plane = lower_base_plane + np.array([0, size[1], 0])

    vertices = np.vstack([lower_base_plane, upper_base_plane])

    faces = np.array(
        quads_to_tris(
            np.array(
                [
                    [0, 1, 2, 3],  # lower base (-y)
                    [7, 6, 5, 4],  # upper base (+y)
                    [4, 5, 1, 0],  # +z face
                    [5, 6, 2, 1],  # +x face
                    [6, 7, 3, 2],  # -z face
                    [3, 7, 4, 0],  # -x face
                ],
                dtype=np.uint64,
            )
        ),
        dtype=np.uint64,
    )

    return _maybe_flatten(vertices, faces, ret_unique_vertices_and_faces)


def create_cube(origin, size, ret_unique_vertices_and_faces=False):
    """
    Tesselate an axis-aligned cube. One vertex is `origin`. The diametrically
    opposite vertex is `size` units along `+x`, `+y`, and `+z`.

    Args:
        origin (np.ndarray): A 3D point vector containing the point on the
            prism with the minimum x, y, and z coords.
        size (float): The length, width, and height of the cube, which should
            be positive.
        ret_unique_vertices_and_faces (bool): When `True` return a vertex
            array containing the unique vertices and an array of faces (i.e.
            vertex indices). When `False`, return a flattened array of
            triangle coordinates.

    Returns:
        object:

        - With `ret_unique_vertices_and_faces=True`: a tuple containing
          an `8x3` array of vertices and a `12x3` array of triangle faces.
        - With `ret_unique_vertices_and_faces=False`: a `12x3x3` matrix of
          flattened triangle coordinates.
    """
    return create_rectangular_prism(
        origin,
        np.repeat(size, 3),
        ret_unique_vertices_and_faces=ret_unique_vertices_and_faces,
    )


def create_triangular_prism(p1, p2, p3, height, ret_unique_vertices_and_faces=False):
    """
    Tesselate a triangular prism whose base is the triangle `p1`, `p2`, `p3`.
    If the vertices are oriented in a counterclockwise direction, the prism
    extends from behind them.

    Args:
        p1 (np.ndarray): A 3D point on the base of the prism.
        p2 (np.ndarray): A 3D point on the base of the prism.
        p3 (np.ndarray): A 3D point on the base of the prism.
        height (float): The height of the prism, which should be positive.
        ret_unique_vertices_and_faces (bool): When `True` return a vertex
            array containing the unique vertices and an array of faces (i.e.
            vertex indices). When `False`, return a flattened array of
            triangle coordinates.

    Returns:
        object:

        - With `ret_unique_vertices_and_faces=True`: a tuple containing
          an `6x3` array of vertices and a `8x3` array of triangle faces.
        - With `ret_unique_vertices_and_faces=False`: a `8x3x3` matrix of
          flattened triangle coordinates.
    """
    from ..plane.plane import Plane

    base_plane = Plane.from_points(p1, p2, p3)
    lower_base_to_upper_base = height * -base_plane.normal
    vertices = np.vstack(([p1, p2, p3], [p1, p2, p3] + lower_base_to_upper_base))

    faces = np.array(
        [
            [0, 1, 2],  # base
            [0, 3, 4],
            [0, 4, 1],  # side 0, 3, 4, 1
            [1, 4, 5],
            [1, 5, 2],  # side 1, 4, 5, 2
            [2, 5, 3],
            [2, 3, 0],  # side 2, 5, 3, 0
            [5, 4, 3],  # base
        ],
        dtype=np.uint64,
    )

    return _maybe_flatten(vertices, faces, ret_unique_vertices_and_faces)


def create_rectangle(ret_unique_vertices_and_faces=False):
    """
    Create a rectangle.

    Args:
        ret_unique_vertices_and_faces (bool): When `True` return a vertex
            array containing the unique vertices and an array of faces (i.e.
            vertex indices). When `False`, return a flattened array of
            triangle coordinates.

    Returns:
        object:

        - With `ret_unique_vertices_and_faces=True`: a tuple containing
          an `4x3` array of vertices and a `2x3` array of triangle faces.
        - With `ret_unique_vertices_and_faces=False`: a `16x3x3` matrix of
          flattened triangle coordinates.
    """
    vertices = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    faces = np.array([[0, 1, 2], [3, 1, 0]], dtype=np.uint64)
    return _maybe_flatten(vertices, faces, ret_unique_vertices_and_faces)
