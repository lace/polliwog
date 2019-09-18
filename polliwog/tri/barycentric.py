import numpy as np
import vg


def compute_barycentric_coordinates(vertices_of_tris, points):
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
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system

    See Also:
        Heidrich, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
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
