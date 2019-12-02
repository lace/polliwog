import numpy as np
import vg
from .._common.shape import columnize


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


def coplanar_points_are_on_same_side_of_line(a, b, p1, p2):
    """
    Test if the given points are on the same side of the given line.

    Args:
        a (np.arraylike): The first 3D point of interest.
        b (np.arraylike): The second 3D point of interest.
        p1 (np.arraylike): A first point which lies on the line of interest.
        p2 (np.arraylike): A second point which lies on the line of interest.

    Returns:
        bool: `True` when `a` and `b` are on the same side of the line defined
        by `p1` and `p2`.
    """
    vg.shape.check(locals(), "a", (3,))
    vg.shape.check(locals(), "b", (3,))
    vg.shape.check(locals(), "p1", (3,))
    vg.shape.check(locals(), "p2", (3,))

    # Uses "same-side technique" from http://blackpawn.com/texts/pointinpoly/default.html
    along_line = b - a
    return vg.dot(vg.cross(along_line, p1 - a), vg.cross(along_line, p2 - a)) >= 0


def contains_coplanar_point(a, b, c, point):
    """
    Assuming `point` is coplanar with the triangle `ABC`, check if it lies
    inside it.
    """
    vg.shape.check(locals(), "a", (3,))
    vg.shape.check(locals(), "b", (3,))
    vg.shape.check(locals(), "c", (3,))
    vg.shape.check(locals(), "point", (3,))

    # Uses "same-side technique" from http://blackpawn.com/texts/pointinpoly/default.html
    return (
        coplanar_points_are_on_same_side_of_line(b, c, point, a)
        and coplanar_points_are_on_same_side_of_line(a, c, point, b)
        and coplanar_points_are_on_same_side_of_line(a, b, point, c)
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
