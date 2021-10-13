import numpy as np
from vg.compat import v2 as vg

__all__ = [
    "rectangular_prism",
    "cube",
    "triangular_prism",
]


def _maybe_flatten(vertices, faces, ret_unique_vertices_and_faces):
    if ret_unique_vertices_and_faces:
        return vertices, faces
    else:
        return vertices[faces]


def rectangular_prism(origin, size, ret_unique_vertices_and_faces=False):
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
    from ..tri import quads_to_tris

    vg.shape.check(locals(), "origin", (3,))
    vg.shape.check(locals(), "size", (3,))

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
                    [4, 5, 1, 0],  # -z face
                    [5, 6, 2, 1],  # +x face
                    [6, 7, 3, 2],  # +z face
                    [3, 7, 4, 0],  # -x face
                ],
            )
        ),
    )

    return _maybe_flatten(vertices, faces, ret_unique_vertices_and_faces)


def cube(origin, size, ret_unique_vertices_and_faces=False):
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
    vg.shape.check(locals(), "origin", (3,))
    if not isinstance(size, float):
        raise ValueError("`size` should be a number")

    return rectangular_prism(
        origin,
        np.repeat(size, 3),
        ret_unique_vertices_and_faces=ret_unique_vertices_and_faces,
    )


def triangular_prism(p1, p2, p3, height, ret_unique_vertices_and_faces=False):
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
    from .. import Plane

    vg.shape.check(locals(), "p1", (3,))
    vg.shape.check(locals(), "p2", (3,))
    vg.shape.check(locals(), "p3", (3,))
    if not isinstance(height, float):
        raise ValueError("`height` should be a number")

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
    )

    return _maybe_flatten(vertices, faces, ret_unique_vertices_and_faces)
