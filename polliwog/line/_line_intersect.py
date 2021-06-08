import numpy as np


def intersect_lines(p0, q0, p1, q1):
    """
    Intersect two lines in 3d: (p0, q0) and (p1, q1). Each should be a 3D
    point.

    See also:
        https://math.stackexchange.com/a/271366/640314
    """
    e = p0 - q0  # direction of line 0
    f = p1 - q1  # direction of line 1

    # Check for special case where we're given the intersection
    # Note that we must check for these, because if p0 == p1 then
    # g would be zero length and we can't continue
    if np.all(p0 == p1) or np.all(p0 == q1):
        return p0
    if np.all(q0 == p1) or np.all(p0 == q1):
        return q0

    g = p0 - p1  # line between to complete a triangle
    h = np.cross(f, g)
    k = np.cross(f, e)
    h_ = np.linalg.norm(h)
    k_ = np.linalg.norm(k)
    if h_ == 0 or k_ == 0:
        # There is no intesection; either parallel (k=0) or collinear (both=0) lines.
        return None

    # Check for the special case of lines in parallel planes.
    # https://math.stackexchange.com/a/697278/640314
    if np.dot(g, k) != 0:
        return None

    l = h_ / k_ * e  # noqa: E741
    sign = -1 if np.all(h / h_ == k / k_) else +1
    return p0 + sign * l


def intersect_2d_lines(p0, q0, p1, q1):
    """
    Intersect two lines: (p0, q0) and (p1, q1). Each should be a 2D
    point.
    """
    # Adapted from http://stackoverflow.com/a/26416320/893113
    dy = q0[1] - p0[1]
    dx = q0[0] - p0[0]
    lhs0 = [-dy, dx]
    rhs0 = p0[1] * dx - dy * p0[0]

    dy = q1[1] - p1[1]
    dx = q1[0] - p1[0]
    lhs1 = [-dy, dx]
    rhs1 = p1[1] * dx - dy * p1[0]

    a = np.array([lhs0, lhs1])
    b = np.array([rhs0, rhs1])

    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None
