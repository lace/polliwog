import numpy as np
import pytest
from .rigid_transform import find_rigid_rotation, find_rigid_transform
from .rotation import euler
from ..box.box import Box


def test_rigid_transform_from_simple_translation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    b = Box(origin=np.array([1.0, 2.0, 3.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = np.eye(3)
    expected_t = np.array([[1.0, 2.0, 3.0]])
    R, t = find_rigid_transform(a, b)
    np.testing.assert_array_almost_equal(a.dot(expected_R) + expected_t, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_transform_from_simple_rotation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    expected_t = np.array([[0, 0, 0]])
    b = a.dot(expected_R) + expected_t
    R, t = find_rigid_transform(a, b)
    np.testing.assert_array_almost_equal(a.dot(expected_R) + expected_t, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_transform_from_rotation_and_translation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    expected_t = np.array([[1, 2, 3]])
    b = a.dot(expected_R) + expected_t
    R, t = find_rigid_transform(a, b)
    np.testing.assert_array_almost_equal(a.dot(expected_R) + expected_t, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_transform_from_rotation_translation_and_scale():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    expected_t = np.array([[1, 2, 3]])
    expected_scale = 1.7
    b = expected_scale * a.dot(expected_R) + expected_t
    s, R, t = find_rigid_transform(a, b, compute_scale=True)
    np.testing.assert_array_almost_equal(s * a.dot(expected_R) + expected_t, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)
    np.testing.assert_almost_equal(s, expected_scale)


def test_rigid_transform_with_reflection():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    b = a * -1
    with pytest.raises(ValueError):
        find_rigid_transform(a, b)


def test_rigid_transform_with_reflection_but_solve_anyway():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    b = a * -1
    expected_R = np.eye(3)
    expected_R[0][0] = expected_R[1][1] = -1
    expected_t = np.array([[0.0, 0.0, -1.0]])
    R, t = find_rigid_transform(a, b, fail_in_degenerate_cases=False)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_rotation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    b = a.dot(expected_R)
    R = find_rigid_rotation(a, b)
    np.testing.assert_array_almost_equal(R, expected_R)


def test_rigid_rotation_single_point():
    a = np.array([[1, 2, 3]])
    expected_R = euler([30, 15, 21])
    b = a.dot(expected_R)
    R = find_rigid_rotation(a, b)
    # in this degenerate case, don't assert that we got the same
    # rotation, just that it is a valid solution
    np.testing.assert_array_almost_equal(a.dot(R), b)


def test_rigid_rotation_with_reflection():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    b = a * -1
    R = find_rigid_rotation(a, b)
    expected_R = np.zeros((3, 3))
    expected_R[0][0] = expected_R[1][2] = expected_R[2][1] = -1
    np.testing.assert_array_almost_equal(R, expected_R)


def test_rigid_rotation_with_scale():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    b = a.dot(expected_R)
    R = find_rigid_rotation(a, b, allow_scaling=True)
    np.testing.assert_array_almost_equal(R, expected_R)
