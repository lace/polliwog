import numpy as np

def inflection_points(points, axis, span):
    '''
    Find the list of vertices that preceed inflection points in a curve. The curve is differentiated
    with respect to the coordinate system defined by axis and span.

    axis: A vector representing the vertical axis of the coordinate system.
    span: A vector representing the the horiztonal axis of the coordinate system.

    returns: a list of points in space corresponding to the vertices that
    immediately preceed inflection points in the curve
    '''

    coords_on_span = points.dot(span)
    dx = np.gradient(coords_on_span)
    coords_on_axis = points.dot(axis)

    # Take the second order finite difference of the curve with respect to the
    # defined coordinate system
    finite_difference_2 = np.gradient(np.gradient(coords_on_axis, dx), dx)

    # Compare the product of all neighboring pairs of points in the second derivative
    # If a pair of points has a negative product, then the second derivative changes sign
    # at one of those points, signalling an inflection point
    is_inflection_point = [finite_difference_2[i] * finite_difference_2[i + 1] <= 0 for i in range(len(finite_difference_2) - 1)]

    inflection_point_indices = [i for i, b in enumerate(is_inflection_point) if b]

    if len(inflection_point_indices) == 0: # pylint: disable=len-as-condition
        return []

    return points[inflection_point_indices]
