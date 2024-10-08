import numpy as np
from polliwog.segment import (
    closest_point_of_line_segment,
    is_point_on_line_segment,
    path_centroid,
    subdivide_segment,
    subdivide_segments,
)
import pytest
from vg.compat import v2 as vg


def test_path_centroid():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.0, 5.0, 0.0]]
    )
    num_edges = 4
    edges = np.vstack([np.arange(num_edges), np.arange(num_edges) + 1]).T
    edges[-1][1] = 0
    segments = vertices[edges]
    np.testing.assert_array_almost_equal(
        path_centroid(segments),
        np.array([0.5, 2.5, 0]),
    )


def test_subdivide_segment_raises_exception_for_invalid_partition_size_type():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(TypeError):
        subdivide_segment(p1, p2, "foobar")


def test_subdivide_segment_raises_exception_for_invalid_partition_size_value():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        subdivide_segment(p1, p2, 1)


def test_subdivide_segment_returns_partition_for_odd_partition_size():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.0, 0.0])

    partition_size = 5

    expected_partition_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_almost_equal(
        subdivide_segment(p1, p2, partition_size), expected_partition_points, decimal=7
    )


def test_subdivide_segment_returns_partition_points_for_even_partition_size():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    partition_size = 6

    expected_partition_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_almost_equal(
        subdivide_segment(p1, p2, partition_size), expected_partition_points, decimal=7
    )


def test_subdivide_segment_returns_partition_omitting_endpoint():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    partition_size = 5

    expected_partition_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_almost_equal(
        subdivide_segment(p1, p2, partition_size, endpoint=False),
        expected_partition_points,
        decimal=7,
    )


def test_subdivide_segments_adds_points_for_equal_length_line_segments():
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 3.0, 0.0],
        ]
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [1.0, 0.4, 0.0],
            [1.0, 0.6, 0.0],
            [1.0, 0.8, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.2, 0.0],
            [1.0, 1.4, 0.0],
            [1.0, 1.6, 0.0],
            [1.0, 1.8, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.2, 0.0],
            [1.0, 2.4, 0.0],
            [1.0, 2.6, 0.0],
            [1.0, 2.8, 0.0],
            [1.0, 3.0, 0.0],
        ]
    )

    np.testing.assert_array_almost_equal(subdivide_segments(v), expected)


def test_subdivide_segments_adds_points_for_nonequal_arbitrarily_oriented_line():
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 2.0, 1.0]])

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.2],
            [0.4, 0.0, 0.4],
            [0.6, 0.0, 0.6],
            [0.8, 0.0, 0.8],
            [1.0, 0.0, 1.0],
            [1.2, 0.0, 1.0],
            [1.4, 0.0, 1.0],
            [1.6, 0.0, 1.0],
            [1.8, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 0.4, 1.0],
            [2.0, 0.8, 1.0],
            [2.0, 1.2, 1.0],
            [2.0, 1.6, 1.0],
            [2.0, 2.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(subdivide_segments(v), expected)


def test_closest_point_of_line_segment():
    # Adapted from public domain algorithm
    # https://gdbooks.gitbooks.io/3dcollisions/content/Chapter1/closest_point_on_line.html
    p1 = np.array([-3.0, -2.0, -1.0])
    p2 = np.array([1.0, 2.0, 3.0])
    segment_vector = p2 - p1

    query_points = np.array(
        [
            [7, -2, -1],
            [1, 5, -5],
            [-7, -10, -4],
            [-4, -7, -8],
            [-6, 5, 7],
            [1, 6, 8],
            [7, 8, 5],
            [5, 5, 3],
        ]
    )

    expected_closest_points = np.array(
        [
            p1 + 10.0 / 12.0 * segment_vector,
            p1 + 7.0 / 12.0 * segment_vector,
            p1,
            p1,
            p2,
            p2,
            p2,
            p2,
        ]
    )
    closest_points = closest_point_of_line_segment(
        points=query_points,
        start_points=np.broadcast_to(p1, query_points.shape),
        segment_vectors=np.broadcast_to(segment_vector, query_points.shape),
    )
    np.testing.assert_array_almost_equal(closest_points, expected_closest_points)

    expected_t_values = np.array([40 / 48, 28 / 48, 0, 0, 1, 1, 1, 1])
    _, t_values = closest_point_of_line_segment(
        points=query_points,
        start_points=np.broadcast_to(p1, query_points.shape),
        segment_vectors=np.broadcast_to(segment_vector, query_points.shape),
        ret_t_values=True,
    )
    np.testing.assert_array_almost_equal(t_values, expected_t_values)


def test_closest_point_of_degenerate_line_segment():
    p = np.array([-3.0, -2.0, -1.0])
    segment_vector = np.zeros(3)

    query_points = np.array(
        [
            [7, -2, -1],
            [1, 5, -5],
            [-7, -10, -4],
            [-4, -7, -8],
            [-6, 5, 7],
            [1, 6, 8],
            [7, 8, 5],
            [5, 5, 3],
        ]
    )

    closest_points, t_values = closest_point_of_line_segment(
        points=query_points,
        start_points=np.broadcast_to(p, query_points.shape),
        segment_vectors=np.broadcast_to(segment_vector, query_points.shape),
        ret_t_values=True,
    )
    np.testing.assert_array_equal(
        closest_points, np.broadcast_to(p, query_points.shape)
    )
    np.testing.assert_array_equal(t_values, np.zeros(len(query_points)))


def test_is_point_on_line_segment():
    start_point = np.zeros(3)
    segment_vector = vg.basis.x
    epsilon = 1e-6

    query_points = np.array(
        [
            # On the line, but not on the line segment.
            [-0.5, 0, 0],
            [-2 * epsilon, 0, 0],
            # On the line within epsilon of the line segment.
            [-epsilon, 0, 0],
            # Within the line segment.
            [0, 0, 0],
            [epsilon, 0, 0],
            [0.5, 0, 0],
            [1 - epsilon, 0, 0],
            [1, 0, 0],
            # On the line within epsilon of the line segment.
            [1 + epsilon, 0, 0],  # This test doesn't work with epsilon < 1e-6.
            # On the line, but not on the line segment.
            [2, 0, 0],
            # Not near the line segment.
            [-0.5, epsilon, 0],
            # A little too far from the line segment.
            [-2 * epsilon, epsilon, 0],
            [-epsilon, epsilon, 0],
            # Within epsilon of the line segment.
            [0, epsilon, 0],
            [epsilon / 2, epsilon / 2, 0],
            [0.5, epsilon, 0],
            [1 - epsilon, epsilon, 0],
            # Within epsilon of the line segment.
            [1, epsilon, 0],
            # A little too far.
            [1 + epsilon, epsilon, 0],
            # Not near the line segment.
            [2, epsilon, 0],
        ]
    )

    np.testing.assert_array_equal(
        is_point_on_line_segment(
            query_points=query_points,
            start_points=np.broadcast_to(start_point, query_points.shape),
            segment_vectors=np.broadcast_to(segment_vector, query_points.shape),
            epsilon=epsilon,
        ),
        np.array(
            [
                # On the line, but not on the line segment.
                False,
                False,
                # On the line within epsilon of the line segment.
                True,
                # Within the line segment.
                True,
                True,
                True,
                True,
                True,
                # On the line within epsilon of the line segment.
                True,
                # On the line, but not on the line segment.
                False,
                # Not near the line segment.
                False,
                # A little too far from the line segment.
                False,
                False,
                # Within epsilon of the line segment.
                True,
                True,
                True,
                True,
                # Within epsilon of the line segment.
                True,
                # A little too far.
                False,
                # Not near the line segment.
                False,
            ]
        ),
    )
