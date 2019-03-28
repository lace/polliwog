import unittest
import numpy as np
from blmath.numerics import vx
from blmath.geometry import Polyline, Plane

class TestPolyline(unittest.TestCase):

    def test_edges(self):
        v = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 2., 0.],
            [1., 3., 0.],
        ])

        expected_open = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
        ])

        np.testing.assert_array_equal(Polyline(v).e, expected_open)

        expected_closed = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 0],
        ])

        np.testing.assert_array_equal(Polyline(v, closed=True).e, expected_closed)

    def test_segments(self):
        polyline = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 2., 0.],
        ]), closed=True)

        expected = np.array([
            [[0., 0., 0.], [1., 0., 0.]],
            [[1., 0., 0.], [1., 1., 0.]],
            [[1., 1., 0.], [1., 2., 0.]],
            [[1., 2., 0.], [0., 0., 0.]],
        ])

        np.testing.assert_array_equal(polyline.segments, expected)

    def test_segment_vectors(self):
        polyline = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 2., 0.],
        ]), closed=True)

        expected = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [-1., -2., 0.],
        ])

        np.testing.assert_array_equal(polyline.segment_vectors, expected)

    def test_length_of_empty_polyline(self):
        polyline = Polyline(None)
        self.assertEqual(polyline.total_length, 0)

        polyline = Polyline(None, closed=True)
        self.assertEqual(polyline.total_length, 0)


    def test_partition_by_length_noop(self):
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 2., 0.],
            [1., 3., 0.],
        ]))

        result = original.copy()
        indices = result.partition_by_length(1., ret_indices=True)

        expected_indices = np.array([0, 1, 2, 3, 4])

        np.testing.assert_array_almost_equal(result.v, original.v)
        np.testing.assert_array_equal(result.e, original.e)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_partition_by_length_degenerate(self):
        '''
        This covers a bug that arose from a numerical stability issue in
        measurement on EC2 / MKL.
        '''
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
        ]))

        result = original.copy()
        indices = result.partition_by_length(1., ret_indices=True)

        expected_indices = np.array([0, 1, 2])

        np.testing.assert_array_almost_equal(result.v, original.v)
        np.testing.assert_array_equal(result.e, original.e)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_partition_by_length_divide_by_two(self):
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 2., 0.],
            [1., 3., 0.],
        ]))

        expected = Polyline(np.array([
            [0., 0., 0.],
            [0.5, 0., 0.],
            [1., 0., 0.],
            [1., 0.5, 0.],
            [1., 1., 0.],
            [1., 1.5, 0.],
            [1., 2., 0.],
            [1., 2.5, 0.],
            [1., 3., 0.],
        ]))

        expected_indices = np.array([0, 2, 4, 6, 8])

        for max_length in (0.99, 0.75, 0.5):
            result = original.copy()
            indices = result.partition_by_length(max_length, ret_indices=True)

            np.testing.assert_array_almost_equal(result.v, expected.v)
            np.testing.assert_array_equal(result.e, expected.e)
            np.testing.assert_array_equal(indices, expected_indices)

    def test_partition_length_divide_by_five(self):
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 2., 0.],
            [1., 3., 0.],
        ]))

        expected = Polyline(np.array([
            [0., 0., 0.],
            [0.2, 0., 0.],
            [0.4, 0., 0.],
            [0.6, 0., 0.],
            [0.8, 0., 0.],
            [1., 0., 0.],
            [1., 0.2, 0.],
            [1., 0.4, 0.],
            [1., 0.6, 0.],
            [1., 0.8, 0.],
            [1., 1., 0.],
            [1., 1.2, 0.],
            [1., 1.4, 0.],
            [1., 1.6, 0.],
            [1., 1.8, 0.],
            [1., 2., 0.],
            [1., 2.2, 0.],
            [1., 2.4, 0.],
            [1., 2.6, 0.],
            [1., 2.8, 0.],
            [1., 3., 0.],
        ]))

        expected_indices = np.array([0, 5, 10, 15, 20])

        for max_length in (0.2, 0.24):
            result = original.copy()
            indices = result.partition_by_length(max_length, ret_indices=True)

            np.testing.assert_array_almost_equal(result.v, expected.v)
            np.testing.assert_array_equal(result.e, expected.e)
            np.testing.assert_array_equal(indices, expected_indices)

    def test_partition_by_length_divide_some_leave_some(self):
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 7., 0.],
            [1., 8., 0.],
        ]))

        expected = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 3., 0.],
            [1., 5., 0.],
            [1., 7., 0.],
            [1., 8., 0.],
        ]))

        expected_indices = np.array([0, 1, 2, 5, 6])

        for max_length in (2., 2.99):
            result = original.copy()
            indices = result.partition_by_length(max_length, ret_indices=True)

            np.testing.assert_array_almost_equal(result.v, expected.v)
            np.testing.assert_array_equal(result.e, expected.e)
            np.testing.assert_array_equal(indices, expected_indices)

    def test_partition_by_length_closed(self):
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 7., 0.],
            [1., 8., 0.],
            [0., 8., 0.],
        ]), closed=True)

        expected = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 3., 0.],
            [1., 5., 0.],
            [1., 7., 0.],
            [1., 8., 0.],
            [0., 8., 0.],
            [0., 6., 0.],
            [0., 4., 0.],
            [0., 2., 0.],
        ]), closed=True)

        expected_indices = np.array([0, 1, 2, 5, 6, 7])

        for max_length in (2., 2.5, 2.6):
            result = original.copy()
            indices = result.partition_by_length(max_length, ret_indices=True)

            np.testing.assert_array_almost_equal(result.v, expected.v)
            np.testing.assert_array_equal(result.e, expected.e)
            np.testing.assert_array_equal(indices, expected_indices)

    def test_flip(self):
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 7., 0.],
            [1., 8., 0.],
            [0., 8., 0.],
        ]), closed=True)

        expected = Polyline(np.array([
            [0., 8., 0.],
            [1., 8., 0.],
            [1., 7., 0.],
            [1., 1., 0.],
            [1., 0., 0.],
            [0., 0., 0.],
        ]), closed=True)

        original.flip()

        np.testing.assert_array_almost_equal(original.v, expected.v)

    def test_reindexed(self):
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 7., 0.],
            [1., 8., 0.],
            [0., 8., 0.],
        ]), closed=True)

        reindexed, edge_mapping = original.reindexed(5, ret_edge_mapping=True)

        expected = Polyline(np.array([
            [0., 8., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 7., 0.],
            [1., 8., 0.],
        ]), closed=True)

        np.testing.assert_array_almost_equal(reindexed.v, expected.v)
        np.testing.assert_array_equal(original.segments[edge_mapping], reindexed.segments)

    def test_cut_by_plane_closed(self):
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 7., 0.],
            [1., 8., 0.],
            [0., 8., 0.],
        ]), closed=True)

        expected = Polyline(np.array([
            [1., 7.5, 0.],
            [1., 8., 0.],
            [0., 8., 0.],
            [0., 7.5, 0.],
        ]), closed=False)
        actual = original.cut_by_plane(
            Plane(point_on_plane=np.array([0., 7.5, 0.]), unit_normal=vx.basis.y))

        np.testing.assert_array_almost_equal(actual.v, expected.v)
        self.assertFalse(actual.closed)

        expected = Polyline(np.array([
            [0., 7.5, 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 7., 0.],
            [1., 7.5, 0.],
        ]), closed=False)
        actual = original.cut_by_plane(
            Plane(point_on_plane=np.array([0., 7.5, 0.]), unit_normal=vx.basis.neg_y))

        np.testing.assert_array_almost_equal(actual.v, expected.v)
        self.assertFalse(actual.closed)

    def test_cut_by_plane_open(self):
        original = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 7., 0.],
            [1., 8., 0.],
        ]), closed=False)

        expected = Polyline(np.array([
            [1., 7.5, 0.],
            [1., 8., 0.],
        ]), closed=False)
        actual = original.cut_by_plane(
            Plane(point_on_plane=np.array([0., 7.5, 0.]), unit_normal=vx.basis.y))

        np.testing.assert_array_almost_equal(actual.v, expected.v)
        self.assertFalse(actual.closed)

        expected = Polyline(np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 7., 0.],
            [1., 7.5, 0.],
        ]), closed=False)
        actual = original.cut_by_plane(
            Plane(point_on_plane=np.array([0., 7.5, 0.]), unit_normal=vx.basis.neg_y))

        np.testing.assert_array_almost_equal(actual.v, expected.v)
        self.assertFalse(actual.closed)
