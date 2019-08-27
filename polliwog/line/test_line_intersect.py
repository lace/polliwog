import numpy as np
from .line_intersect import line_intersect, line_intersect3


def test_line_intersect():
    p0, q0 = np.array([[0.0, 3.0], [4.0, 11.0]])
    p1, q1 = np.array([[-2.0, 8.0], [6.0, 4.0]])
    result = line_intersect(p0, q0, p1, q1)
    np.testing.assert_array_equal(result, [1.6, 6.2])


def test_line_intersect_duplicate_point():
    p0, q0 = np.array([[0.0, 3.0], [5.0, 5.0]])
    p1, q1 = np.array([[5.0, 5.0], [6.0, 4.0]])
    result = line_intersect(p0, q0, p1, q1)
    np.testing.assert_array_equal(result, [5.0, 5.0])


def test_line_intersect3_with_colinear_lines():
    p0, q0 = np.array([[0.0, 1.0, 2.0], [0.0, 10.0, 20.0]])
    p1, q1 = np.array([[0.0, 2.0, 4.0], [0.0, 4.0, 8.0]])
    result = line_intersect3(p0, q0, p1, q1)
    assert result is None


def test_line_intersect3_with_parallel_lines():
    p0, q0 = np.array([[0.0, 1.0, 2.0], [0.0, 10.0, 20.0]])
    p1, q1 = np.array([[1.0, 2.0, 3.0], [1.0, 11.0, 21.0]])
    result = line_intersect3(p0, q0, p1, q1)
    assert result is None


def test_line_intersect3_with_degenerate_input_p():
    p0, q0 = np.array([[0.0, 1.0, 2.0], [0.0, 10.0, 20.0]])
    p1, q1 = np.array([[0.0, 1.0, 2.0], [1.0, 11.0, 21.0]])
    result = line_intersect3(p0, q0, p1, q1)
    np.testing.assert_array_equal(result, [0.0, 1.0, 2.0])


def test_line_intersect3_with_degenerate_input_q():
    p0, q0 = np.array([[0.0, 1.0, 2.0], [0.0, 10.0, 20.0]])
    p1, q1 = np.array([[1.0, 2.0, 3.0], [0.0, 10.0, 20.0]])
    result = line_intersect3(p0, q0, p1, q1)
    np.testing.assert_array_equal(result, [0.0, 10.0, 20.0])


def test_line_intersect3_with_degenerate_input_q_2():
    p0, q0 = np.array([[0.0, 1.0, 2.0], [0.0, 10.0, 20.0]])
    p1, q1 = np.array([[0.0, 10.0, 20.0], [1.0, 2.0, 3.0]])
    result = line_intersect3(p0, q0, p1, q1)
    np.testing.assert_array_equal(result, [0.0, 10.0, 20.0])


def test_line_intersect3_example_1():
    # This example tests the codirectional cross product case
    p0, q0 = np.array([[5.0, 5.0, 4.0], [10.0, 10.0, 6.0]])
    p1, q1 = np.array([[5.0, 5.0, 5.0], [10.0, 10.0, 3.0]])
    result = line_intersect3(p0, q0, p1, q1)
    np.testing.assert_array_equal(result, [25.0 / 4, 25.0 / 4, 9.0 / 2])


def test_line_intersect3_example_2():
    # This example tests the opposite direction cross product case
    p0, q0 = np.array([[5.0, 5.0, 4.0], [10.0, 10.0, -6.0]])
    p1, q1 = np.array([[5.0, 5.0, 5.0], [10.0, 10.0, -3.0]])
    result = line_intersect3(p0, q0, p1, q1)
    np.testing.assert_array_equal(result, [2.5, 2.5, 9])


def test_line_intersect3_example_3():
    p0, q0 = np.array([[6.0, 8.0, 4.0], [12.0, 15.0, 4.0]])
    p1, q1 = np.array([[6.0, 8.0, 2.0], [12.0, 15.0, 6.0]])
    result = line_intersect3(p0, q0, p1, q1)
    np.testing.assert_array_equal(result, [9.0, 23.0 / 2, 4.0])
