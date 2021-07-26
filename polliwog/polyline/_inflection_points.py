import numpy as np
from vg.compat import v2 as vg


def inflection_points(points, rise_axis, run_axis):
    """
    Find the list of vertices that preceed inflection points in a curve. The
    curve is differentiated with respect to the coordinate system defined by
    `rise_axis` and `run_axis`.

    Interestingly, `lambda x: 2*x + 1` should have no inflection points, but
    almost every point on the line is detected. It's because a zero or zero
    crossing in the second derivative is necessary but not sufficient to
    detect an inflection point. You also need a higher derivative of odd
    order that's non-zero. But that gets ugly to detect reliably using sparse
    finite differences. Just know that if you've got a straight line this
    method will go a bit haywire.

    rise_axis: A vector representing the vertical axis of the coordinate system.
    run_axis: A vector representing the the horiztonal axis of the coordinate system.

    returns: a list of points in space corresponding to the vertices that
    immediately preceed inflection points in the curve
    """
    vg.shape.check(locals(), "points", (-1, 3))
    vg.shape.check(locals(), "rise_axis", (3,))
    vg.shape.check(locals(), "run_axis", (3,))

    coords_on_run_axis = points.dot(run_axis)
    coords_on_rise_axis = points.dot(rise_axis)

    # Take the second order finite difference of the curve with respect to the
    # defined coordinate system
    finite_difference_1 = np.gradient(coords_on_rise_axis, coords_on_run_axis)
    finite_difference_2 = np.gradient(finite_difference_1, coords_on_run_axis)

    # Compare the product of all neighboring pairs of points in the second
    # derivative. If a pair of points has a negative product, then the second
    # derivative changes sign between those points. These are the inflection
    # points.
    is_inflection_point = np.concatenate(
        [finite_difference_2[:-1] * finite_difference_2[1:] <= 0, [False]]
    )

    return points[is_inflection_point]


def point_of_max_acceleration(points, rise_axis, run_axis, subdivide_by_length=None):
    """
    Find the point on a curve where the curve is maximally accelerating
    in the direction specified by `rise_axis`. `run_axis` is the horizontal
    axis along which slices are taken.

    Args:
        points (np.arraylike): A stack of points, as `kx3`. For best
            results, trim these to the area of interest before calling.
        rise_axis (np.arraylike): The vertical axis, as a 3D vector.
        run_axis (np.arraylike): The horizonal axis, as a 3D vector.
        subdivide_by_length (float): When provided, the maximum space
            between each point. The idea is keep the slice width small,
            however this constraint is applied in 3D space, not along
            the `run_axis`. For best results pass a value that is small
            relative to the changes in the geometry. When `None`, the
            points are used without modification.
    """
    from ..polyline._polyline_object import Polyline

    k = vg.shape.check(locals(), "points", (-1, 3))
    vg.shape.check(locals(), "rise_axis", (3,))
    vg.shape.check(locals(), "run_axis", (3,))

    if k < 2:
        raise ValueError("At least two points are required")

    if subdivide_by_length is not None:
        subdivided = Polyline(v=points, is_closed=False).subdivided_by_length(
            subdivide_by_length
        )
        points = subdivided.v

    coords_on_run_axis = points.dot(run_axis)
    coords_on_rise_axis = points.dot(rise_axis)

    finite_difference_1 = np.gradient(coords_on_rise_axis, coords_on_run_axis)
    finite_difference_2 = np.gradient(finite_difference_1, coords_on_run_axis)

    # `np.argmax(finite_difference_2)` produces false positives where the first
    # derivative of the next point is positive. Exclude these bogus points.
    valid_points = np.logical_and(
        np.roll(finite_difference_1, 1) > 0,
        np.roll(finite_difference_1, -1) > 0,
    )
    valid_points[0] = False
    valid_points[-1] = False

    try:
        index = np.argmax(finite_difference_2[valid_points])
    except ValueError:
        return None

    return points[valid_points][index]
