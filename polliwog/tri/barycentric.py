import numpy as np

def barycentric_coordinates_of_projection(p, q, u, v):
    """ Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.

    See Heidrich, Computing the Barycentric Coordinates of a Projected Point, JGT 05
    at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf

    Args:
        p: point to project
        q: a vertex of the triangle to project into
        u,v: edges of the the triangle such that it has vertices q, q+u, q+v

    Returns:
        b: barycentric coordinates of p's projection in triangle defined by q,u,v
            vectorized so p,q,u,v can all be 3xN
    """

    p = p.T
    q = q.T
    u = u.T
    v = v.T

    n = np.cross(u, v, axis=0)
    s = np.sum(n*n, axis=0)

    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    if np.isscalar(s):
        s = s if s else np.spacing(1)
    else:
        s[s == 0] = np.spacing(1)

    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = np.sum(np.cross(u, w, axis=0) * n, axis=0) * oneOver4ASquared
    b1 = np.sum(np.cross(w, v, axis=0) * n, axis=0) * oneOver4ASquared
    b = np.vstack((1 - b1 - b2, b1, b2))

    return b.T
