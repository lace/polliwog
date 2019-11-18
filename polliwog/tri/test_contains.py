import numpy as np
from .contains import contains_coplanar_point


def test_contains_coplanar_point():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([4.0, 0.1, 0.0])
    c = np.array([3.0, 3.1, 0.0])

    # Not sure why, but `is True` does not work.
    assert contains_coplanar_point(a, b, c, a) == True  # noqa: E712
    assert contains_coplanar_point(a, b, c, b) == True  # noqa: E712
    assert contains_coplanar_point(a, b, c, c) == True  # noqa: E712
    assert (
        contains_coplanar_point(a, b, c, np.array([2.0, 1.0, 0.0])) == True
    )  # noqa: E712

    # Unexpected, as it's not in the plane, though if projected to the plane,
    # it is in the triangle.
    assert (
        contains_coplanar_point(a, b, c, np.array([0.0, 0.0, 1.0])) == True
    )  # noqa: E712

    assert (
        contains_coplanar_point(a, b, c, np.array([2.0, 0.0, 0.0])) == False
    )  # noqa: E712
    assert (
        contains_coplanar_point(a, b, c, np.array([2.0, 5.0, 0.0])) == False
    )  # noqa: E712
