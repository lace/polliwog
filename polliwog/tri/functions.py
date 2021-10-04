import numpy as np
from vg.compat import v2 as vg
from .._common.shape import check_shape_any, columnize
from ..line._line_functions import coplanar_points_are_on_same_side_of_line

FACE_DTYPE = np.int64

__all__ = [
    "FACE_DTYPE",
    "edges_of_faces",
    "surface_normals",
    "tri_contains_coplanar_point",
    "barycentric_coordinates_of_points",
]


def edges_of_faces(faces, normalize=True):
    """
    Given a stack of triangles expressed as vertex indices, return a
    normalized array of edges. When `normalize=True`, sort the edges so they
    more easily can be compared.
    """
    vg.shape.check(locals(), "faces", (-1, 3))
    assert faces.dtype == FACE_DTYPE

    # TODO: It's probably possible to accomplish this more efficiently. Maybe
    # with `np.pick()`?
    interleaved_edges = np.stack(
        [faces[:, 0:2], faces[:, 1:3], np.roll(faces, 1, axis=1)[:, 0:2]]
    )
    flattened_edges = np.swapaxes(interleaved_edges, 0, 1).reshape(-1, 2)
    return np.sort(flattened_edges, axis=1) if normalize else flattened_edges


def surface_normals(points, normalize=True):
    """
    Compute the surface normal of a triangle. The direction of the normal
    follows conventional counter-clockwise winding and the right-hand
    rule.

    Also works on stacked inputs (i.e. many sets of three points).
    """
    points, _, transform_result = columnize(points, (-1, 3, 3), name="points")

    p1s = points[:, 0]
    p2s = points[:, 1]
    p3s = points[:, 2]
    v1s = p2s - p1s
    v2s = p3s - p1s
    normals = vg.cross(v1s, v2s)

    if normalize:
        normals = vg.normalize(normals)

    return transform_result(normals)


def tri_contains_coplanar_point(a, b, c, point):
    """
    Assuming `point` is coplanar with the triangle `ABC`, check if it lies
    inside it.
    """
    check_shape_any(a, (3,), (-1, 3), name="a")
    vg.shape.check(locals(), "b", a.shape)
    vg.shape.check(locals(), "c", a.shape)
    vg.shape.check(locals(), "point", a.shape)

    # Uses "same-side technique" from http://blackpawn.com/texts/pointinpoly/default.html
    return np.logical_and(
        np.logical_and(
            coplanar_points_are_on_same_side_of_line(b, c, point, a),
            coplanar_points_are_on_same_side_of_line(a, c, point, b),
        ),
        coplanar_points_are_on_same_side_of_line(a, b, point, c),
    )


def barycentric_coordinates_of_points(vertices_of_tris, points):
    """
    Compute barycentric coordinates for the projection of a set of points to a
    given set of triangles specfied by their vertices.

    These barycentric coordinates can refer to points outside the triangle.
    This happens when one of the coordinates is negative. However they can't
    specify points outside the triangle's plane. (That requires tetrahedral
    coordinates.)

    The returned coordinates supply a linear combination which, applied to the
    vertices, returns the projection of the original point the plane of the
    triangle.

    Args:
        vertices_of_tris (np.arraylike): A set of triangle vertices as `kx3x3`.
        points (np.arraylike): Coordinates of points as `kx3`.

    Returns:
        np.ndarray: Barycentric coordinates as `kx3`

    See Also:
        - https://en.wikipedia.org/wiki/Barycentric_coordinate_system
        - Heidrich, "Computing the Barycentric Coordinates of a Projected
          Point," JGT 05 (http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf)
    """
    k = vg.shape.check(locals(), "vertices_of_tris", (-1, 3, 3))
    vg.shape.check(locals(), "points", (k, 3))

    p = points.T
    q = vertices_of_tris[:, 0].T
    u = (vertices_of_tris[:, 1] - vertices_of_tris[:, 0]).T
    v = (vertices_of_tris[:, 2] - vertices_of_tris[:, 0]).T

    n = np.cross(u, v, axis=0)
    s = np.sum(n * n, axis=0)

    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = np.spacing(1)

    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = np.sum(np.cross(u, w, axis=0) * n, axis=0) * oneOver4ASquared
    b1 = np.sum(np.cross(w, v, axis=0) * n, axis=0) * oneOver4ASquared
    b = np.vstack((1 - b1 - b2, b1, b2))

    return b.T
