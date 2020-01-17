import numpy as np
import vg


def percentile(points, axis, percentile):
    """
    Given a cloud of points and an axis, find a point along that axis
    from the centroid at the given percentile.

    Args:
        points (np.arraylike): A `kx3` stack of points.
        axis (np.arraylike): A 3D vector specifying the direction of
            interest.
        percentile (float): The desired percentile.

    Returns:
        np.ndarray: A 3D point at the requested percentile.
    """
    k = vg.shape.check(locals(), "points", (-1, 3))
    if k < 1:
        raise ValueError("At least one point is needed")
    vg.shape.check(locals(), "axis", (3,))
    if vg.almost_zero(axis):
        raise ValueError("Axis must be non-zero")

    axis = vg.normalize(axis)
    coords_on_axis = points.dot(axis)
    selected_coord_on_axis = np.percentile(coords_on_axis, percentile)
    centroid = np.average(points, axis=0)
    return vg.reject(centroid, axis) + selected_coord_on_axis * axis
