import numpy as np
import vg


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


def point_of_max_acceleration(points, axis, span, span_spacing=None, plot=False):
    """
    Find the point on a curve where the curve is maximally accelerating
    in the direction specified by `axis`.

    `span` is the horizontal axis along which slices are taken, and
    `span_spacing`, if provided, indices an upper bound on the size
    of each slice.

    For best results, slice the points first into a short section that
    spans the area of interest.

    Args:
        points (np.arraylike): A stack of points, as `kx3`.
        axis (np.arraylike): The vertical axis, as a 3D vector.
        span (np.arraylike): The horizonal axis, as a 3D vector.
        span_spacing (float): When provided, the maximum width of each
            slice. For best results pass a value that is small relative to
            the changes in the geometry. When `None`, the original points
            are used.
    """
    from ..polyline._polyline_object import Polyline

    vg.shape.check(locals(), "points", (-1, 3))
    vg.shape.check(locals(), "axis", (3,))
    vg.shape.check(locals(), "span", (3,))

    if span_spacing is not None:
        points = (
            Polyline(v=points, is_closed=False).subdivided_by_length(span_spacing).v
        )

    coords_on_span = points.dot(span)
    coords_on_axis = points.dot(axis)

    finite_difference_1 = np.gradient(coords_on_axis, coords_on_span)
    finite_difference_2 = np.gradient(finite_difference_1, coords_on_span)

    # def moving_average(x, w):
    #     return np.convolve(x, np.ones(w), 'valid') / w

    # window = 2000
    # moving_average_1 = moving_average(finite_difference_1, window)
    # (zero_crossings,) = np.where(np.diff(np.sign(moving_average_1)))
    # index = zero_crossings[-1] - int(window / 2.0)

    # When there are no zero crossings, this is probably because there is no
    # inflection point, such as on the side of the bust. Use the first point
    # instead.
    # try:
    #     index = zero_crossings[zero_crossings < np.argmin(moving_average_1)][-1]
    # except IndexError:
    #     index = 0
    #     # raise ValueError("No inflection point found")

    # Do not choose a point where the first derivate of the next point is negative.
    # valid_points = np.concatenate([(finite_difference_1 > 0)[1:], [False]])
    valid_points = np.logical_and(
        np.roll(finite_difference_1, 1) > 0, np.roll(finite_difference_1, -1) > 0,
    )
    valid_points[0] = False
    valid_points[-1] = False

    try:
        index = np.argmax(finite_difference_2[valid_points])
    except ValueError:
        # import pdb; pdb.set_trace()
        plot = True
        index = None
    # index = np.argmax(finite_difference_2)

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="darkgrid")

        # sns.relplot(data=np.hstack([xs, np.arange(len(xs))]))

        fig, axs = plt.subplots(3)

        import pdb

        pdb.set_trace()
        xs = coords_on_span
        axs[0].plot(xs, coords_on_axis, label="finite1")
        axs[1].scatter(
            xs[valid_points], finite_difference_1[valid_points], label="finite1"
        )
        axs[1].plot(xs, finite_difference_1, label="finite1")
        axs[2].plot(xs, finite_difference_2, label="finite1")
        axs[2].scatter(
            xs[valid_points], finite_difference_2[valid_points], label="finite1"
        )

        # axs[3].plot(
        #     xs, np.gradient(finite_difference_2, coords_on_span), label="finite1"
        # )
        # axs[1].plot(xs[: -window + 1], moving_average_1, label="finite1")
        # axs[5].plot(
        #     xs[: -window + 1],
        #     moving_average(finite_difference_2, window),
        #     label="finite1",
        # )
        # axs[0].add_artist(
        #     plt.Circle((xs[index], coords_on_axis[index]), 0.1, fill=False, color="red")
        # )
        # for coord in zero_crossings:
        #     axs[1].add_artist(
        #         plt.Circle(
        #             (xs[coord], moving_average_1[coord]), 0.1, fill=False, color="red"
        #         )
        #     )
        # axs[2].plot(
        #     xs[int(window / 2) : -int(window / 2) + 1],
        #     np.gradient(
        #         moving_average_1, coords_on_span[int(window / 2) : -int(window / 2) + 1]
        #     ),
        #     label="finite1",
        # )
        # axs[3].plot(xs, finite_difference_1, label='finite1')
        plt.show()

    # if plot:
    #     return None

    # import pdb; pdb.set_trace()

    # threshold = 0.8 * np.amax(finite_difference_1)
    # (peaks,) = (finite_difference_2 > threshold).nonzero()
    # index = int(np.average(peaks))

    if index is None:
        return None

    return points[valid_points][index]
    # return points[index]
