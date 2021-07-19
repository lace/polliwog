import numpy as np
from vg.compat import v2 as vg


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


def extent(points, ret_indices=False):
    """
    Find the distance between the two farthest-most points.

    Args:
        points (np.arraylike): A `kx3` stack of points.
        ret_indices (bool): When `True`, return the indices along with the
            distance.

    Returns:
        object: With `ret_indices=False`, the distance; with
        `ret_indices=True` a tuple `(distance, first_index, second_index)`.

    Note:
        This is implemented using a brute-force method.
    """
    k = vg.shape.check(locals(), "points", (-1, 3))
    if k < 2:
        raise ValueError("At least two points are required")

    farthest_i = -1
    farthest_j = -1
    farthest_distance = -1
    for i, probe in enumerate(points):
        distances = vg.euclidean_distance(points, probe)
        this_farthest_j = np.argmax(distances)
        if distances[this_farthest_j] > farthest_distance:
            farthest_i = i
            farthest_j = this_farthest_j
            farthest_distance = distances[this_farthest_j]
    if ret_indices:
        return farthest_distance, farthest_i, farthest_j
    else:
        return farthest_distance
