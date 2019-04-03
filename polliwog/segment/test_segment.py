import pytest
import numpy as np
from .segment import partition, partition_segment, partition_segment_old


def test_partition_segment_old_raises_exception_for_invalid_partition_size_type():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(TypeError):
        partition_segment_old(p1, p2, "foobar")


def test_partition_segment_old_raises_exception_for_invalid_partition_size_value():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        partition_segment_old(p1, p2, 1)


def test_partition_segment_old_returns_partition_for_odd_partition_size():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.0, 0.0])

    partition_size = 4

    expected_partition_points = np.array(
        [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    )

    np.testing.assert_array_almost_equal(
        partition_segment_old(p1, p2, partition_size),
        expected_partition_points,
        decimal=7,
    )


def test_partition_segment_old_returns_partition_points_for_even_partition_size():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    partition_size = 5

    expected_partition_points = np.array(
        [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0], [0.8, 0.0, 0.0]]
    )

    np.testing.assert_array_almost_equal(
        partition_segment_old(p1, p2, partition_size),
        expected_partition_points,
        decimal=7,
    )


def test_partition_segment_old_returns_partition_points_in_oriented_order():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    partition_size = 5

    expected_partition_points = np.array(
        [[0.8, 0.0, 0.0], [0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )

    np.testing.assert_array_almost_equal(
        partition_segment_old(p2, p1, partition_size),
        expected_partition_points,
        decimal=7,
    )


def test_partition_segment_old_returns_partition_points_for_diagonal_segment():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 1.0, 0.0])

    partition_size = 3

    dist = np.linalg.norm(p2 - p1)
    domain = [(1 / 3.0) * dist, (2 / 3.0) * dist]

    unit_direction = (p2 - p1) / dist

    expected_partition_points = np.array(
        [p1 + scalar * unit_direction for scalar in domain]
    )

    np.testing.assert_array_almost_equal(
        partition_segment_old(p1, p2, partition_size),
        expected_partition_points,
        decimal=7,
    )


def test_partition_segment_raises_exception_for_invalid_partition_size_type():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(TypeError):
        partition_segment(p1, p2, "foobar")


def test_partition_segment_raises_exception_for_invalid_partition_size_value():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        partition_segment(p1, p2, 1)


def test_partition_segment_returns_partition_for_odd_partition_size():
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
        partition_segment(p1, p2, partition_size), expected_partition_points, decimal=7
    )


def test_partition_segment_returns_partition_points_for_even_partition_size():
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
        partition_segment(p1, p2, partition_size), expected_partition_points, decimal=7
    )


def test_partition_segment_returns_partition_omitting_endpoint():
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
        partition_segment(p1, p2, partition_size, endpoint=False),
        expected_partition_points,
        decimal=7,
    )


def test_partition_adds_points_for_equal_length_line_segments():
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

    np.testing.assert_array_almost_equal(partition(v), expected)


def test_partition_adds_points_for_nonequal_arbitrarily_oriented_line():
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

    np.testing.assert_array_almost_equal(partition(v), expected)
