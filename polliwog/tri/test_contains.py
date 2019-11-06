import numpy as np
from .contains import contains_coplanar_point


def test_contains_coplanar_point():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([4.0, 0.1, 0.0])
    c = np.array([3.0, 3.1, 0.0])

    assert contains_coplanar_point(a, b, c, a) is True
    assert contains_coplanar_point(a, b, c, b) is True
    assert contains_coplanar_point(a, b, c, c) is True
    assert contains_coplanar_point(a, b, c, np.array([2.0, 1.0, 0.0])) is True

    # Unexpected, as it's not in the plane, though if projected to the plane,
    # it is in the triangle.
    assert contains_coplanar_point(a, b, c, np.array([0.0, 0.0, 1.0])) is True

    assert contains_coplanar_point(a, b, c, np.array([2.0, 0.0, 0.0])) is False
    assert contains_coplanar_point(a, b, c, np.array([2.0, 5.0, 0.0])) is False
