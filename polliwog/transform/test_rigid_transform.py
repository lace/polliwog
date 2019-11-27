import numpy as np
from .rigid_transform import find_rigid_rotation, find_rigid_transform
from .rotation import euler
from ..box.box import Box


def test_rigid_transform_from_simple_translation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v.T
    b = Box(origin=np.array([1.0, 2.0, 3.0]), size=np.array([1.0, 1.0, 1.0])).v.T
    expected_R = np.eye(3)
    expected_t = np.array([[1.0, 2.0, 3.0]]).T
    R, t = find_rigid_transform(a, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_transform_from_simple_rotation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v.T
    expected_R = euler([30, 15, 21])
    expected_t = np.array([[0, 0, 0]]).T
    b = expected_R.dot(a) + expected_t
    R, t = find_rigid_transform(a, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_transform_from_rotation_and_translation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v.T
    expected_R = euler([30, 15, 21])
    expected_t = np.array([[1, 2, 3]]).T
    b = expected_R.dot(a) + expected_t
    R, t = find_rigid_transform(a, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_transform_from_rotation_translation_and_scale():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v.T
    expected_R = euler([30, 15, 21])
    expected_t = np.array([[1, 2, 3]]).T
    expected_scale = 1.7
    b = expected_scale * expected_R.dot(a) + expected_t
    s, R, t = find_rigid_transform(a, b, compute_scale=True)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)
    np.testing.assert_almost_equal(s, expected_scale)


def test_rigid_rotation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v.T
    expected_R = euler([30, 15, 21])
    b = expected_R.dot(a)
    R = find_rigid_rotation(a, b)
    np.testing.assert_array_almost_equal(R, expected_R)
