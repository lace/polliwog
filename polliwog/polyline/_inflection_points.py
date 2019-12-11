import numpy as np
import vg


def inflection_points(points, axis, span):
    """
    Find the list of vertices that preceed inflection points in a curve. The curve is differentiated
    with respect to the coordinate system defined by axis and span.

    Interestingly, `lambda x: 2*x + 1` should have no inflection points, but
    almost every point on the line is detected. It's because a zero or zero
    crossing in the second derivative is necessary but not sufficient to
    detect an inflection point. You also need a higher derivative of odd
    order that's non-zero. But that gets ugly to detect reliably using sparse
    finite differences. Just know that if you've got a straight line this
    method will go a bit haywire.

    axis: A vector representing the vertical axis of the coordinate system.
    span: A vector representing the the horiztonal axis of the coordinate system.

    returns: a list of points in space corresponding to the vertices that
    immediately preceed inflection points in the curve
    """
    vg.shape.check(locals(), "points", (-1, 3))
    vg.shape.check(locals(), "axis", (3,))
    vg.shape.check(locals(), "span", (3,))

    coords_on_span = points.dot(span)
    coords_on_axis = points.dot(axis)

    # Take the second order finite difference of the curve with respect to the
    # defined coordinate system
    finite_difference_1 = np.gradient(coords_on_axis, coords_on_span)
    finite_difference_2 = np.gradient(finite_difference_1, coords_on_span)

    # Compare the product of all neighboring pairs of points in the second
    # derivative. If a pair of points has a negative product, then the second
    # derivative changes sign between those points. These are the inflection
    # points.
    is_inflection_point = np.concatenate(
        [finite_difference_2[:-1] * finite_difference_2[1:] <= 0, [False]]
    )

    return points[is_inflection_point]
