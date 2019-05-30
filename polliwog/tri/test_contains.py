import numpy as np
from .contains import contains_coplanar_point


def test_contains():
    triangle = np.array([[0.0, 0.0, 0.0], [4.0, 0.1, 0.0], [3.0, 3.1, 0.0]])

    assert contains_coplanar_point(*triangle, triangle[0]) == True
    assert contains_coplanar_point(*triangle, triangle[1]) == True
    assert contains_coplanar_point(*triangle, triangle[2]) == True
    assert contains_coplanar_point(*triangle, np.array([2.0, 1.0, 0.0])) == True

    # Unexpected, as it's not in the plane, though if projected to the plane,
    # it is in the triangle.
    assert contains_coplanar_point(*triangle, np.array([0.0, 0.0, 1.0])) == True

    assert contains_coplanar_point(*triangle, np.array([2.0, 0.0, 0.0])) == False
    assert contains_coplanar_point(*triangle, np.array([2.0, 5.0, 0.0])) == False
