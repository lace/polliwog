import numpy as np


def _maybe_flatten(vertices, faces, ret_unique_vertices_and_faces):
    if ret_unique_vertices_and_faces:
        return vertices, faces
    else:
        return vertices[faces]


def create_rectangular_prism(origin, size, ret_unique_vertices_and_faces=False):
    """
    Return vertices (or unique verties and faces) of an axis-aligned
    rectangular prism. One vertex is `origin`; the diametrically opposite
    vertex is `origin + size`.

    size: 3x1 array.

    """
    from .arity import quads_to_tris

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
    Return vertices (or unique verties and faces) with an axis-aligned cube.
    One vertex is `origin`; the diametrically opposite vertex is `size` units
    along +x, +y, and +z.

    size: int or float.

    """
    return create_rectangular_prism(
        origin,
        np.repeat(size, 3),
        ret_unique_vertices_and_faces=ret_unique_vertices_and_faces,
    )


def create_triangular_prism(p1, p2, p3, height, ret_unique_vertices_and_faces=False):
    """
    Return vertices (or unique verties and faces) of a triangular prism whose
    base is the triangle p1, p2, p3. If the vertices are oriented in a
    counterclockwise direction, the prism extends from behind them.

    Imported from lace.
    """
    from ..plane.plane import Plane

    base_plane = Plane.from_points(p1, p2, p3)
    lower_base_to_upper_base = (
        height * -base_plane.normal
    )  # pylint: disable=invalid-unary-operand-type
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


def create_horizontal_plane(ret_unique_vertices_and_faces=False):
    """
    Creates a horizontal plane.
    """
    vertices = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    faces = np.array([[0, 1, 2], [3, 1, 0]], dtype=np.uint64)
    return _maybe_flatten(vertices, faces, ret_unique_vertices_and_faces)
