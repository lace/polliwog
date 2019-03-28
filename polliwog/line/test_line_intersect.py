import unittest
import numpy as np

class TestLineIntersect(unittest.TestCase):

    def test_line_intersect(self):
        from blmath.geometry.segment import line_intersect
        p0, q0 = np.array([[0., 3.], [4., 11.]])
        p1, q1 = np.array([[-2., 8.], [6., 4.]])
        result = line_intersect(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [1.6, 6.2])

    def test_line_intersect_duplicate_point(self):
        from blmath.geometry.segment import line_intersect
        p0, q0 = np.array([[0., 3.], [5., 5.]])
        p1, q1 = np.array([[5., 5.], [6., 4.]])
        result = line_intersect(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [5., 5.])


class TestLineIntersect3D(unittest.TestCase):

    def test_line_intersect3_with_colinear_lines(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[0., 1., 2.], [0., 10., 20.]])
        p1, q1 = np.array([[0., 2., 4.], [0., 4., 8.]])
        result = line_intersect3(p0, q0, p1, q1)
        self.assertIsNone(result)

    def test_line_intersect3_with_parallel_lines(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[0., 1., 2.], [0., 10., 20.]])
        p1, q1 = np.array([[1., 2., 3.], [1., 11., 21.]])
        result = line_intersect3(p0, q0, p1, q1)
        self.assertIsNone(result)

    def test_line_intersect3_with_degenerate_input_p(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[0., 1., 2.], [0., 10., 20.]])
        p1, q1 = np.array([[0., 1., 2.], [1., 11., 21.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [0., 1., 2.])

    def test_line_intersect3_with_degenerate_input_q(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[0., 1., 2.], [0., 10., 20.]])
        p1, q1 = np.array([[1., 2., 3.], [0., 10., 20.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [0., 10., 20.])

    def test_line_intersect3_example_1(self):
        # This example tests the codirectional cross product case
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[5., 5., 4.], [10., 10., 6.]])
        p1, q1 = np.array([[5., 5., 5.], [10., 10., 3.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [25./4, 25./4, 9./2])

    def test_line_intersect3_example_2(self):
        # This example tests the opposite direction cross product case
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[5., 5., 4.], [10., 10., -6.]])
        p1, q1 = np.array([[5., 5., 5.], [10., 10., -3.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [2.5, 2.5, 9])

    def test_line_intersect3_example_3(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[6., 8., 4.], [12., 15., 4.]])
        p1, q1 = np.array([[6., 8., 2.], [12., 15., 6.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [9., 23./2, 4.])
