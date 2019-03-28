import numpy as np
import vg


def surface_normal(points, normalize=True):
    """
    Compute the surface normal of a triangle.

    Can provide three points or an array of points.
    """
    p1 = points[..., 0, :]
    p2 = points[..., 1, :]
    p3 = points[..., 2, :]
    normal = np.cross(p2 - p1, p3 - p1)
    return vg.normalize(normal) if normalize else normal
