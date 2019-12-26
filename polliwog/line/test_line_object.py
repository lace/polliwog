import numpy as np
import pytest
import vg
from ._line_object import Line


def test_constructor():
    points = np.array([[5.0, 5.0, 4.0], [10.0, 10.0, 6.0]])
    line = Line.from_points(*points)
    np.testing.assert_array_equal(line.reference_point, points[0])
    np.testing.assert_array_equal(line.reference_points, points)


def test_constructor_error():
    with pytest.raises(ValueError):
        Line.from_points(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        Line.from_points(np.array([1.0, 2.0]), np.array([3.0, 4.0]))


def test_intersect():
    # This example intersection point came from https://math.stackexchange.com/a/270991/640314
    line1 = Line.from_points(np.array([5.0, 5.0, 4.0]), np.array([10.0, 10.0, 6.0]))
    line2 = Line.from_points(np.array([5.0, 5.0, 5.0]), np.array([10.0, 10.0, 3.0]))

    expected_intersection_point = np.array([6.25, 6.25, 4.5])

    np.testing.assert_array_almost_equal(
        line1.intersect_line(line2), expected_intersection_point
    )


def test_project():
    p1 = np.array([5.0, 5.0, 4.0])
    p2 = np.array([10.0, 10.0, 6.0])
    line = Line.from_points(p1, p2)

    np.testing.assert_array_almost_equal(line.project(p1), p1)
    np.testing.assert_array_almost_equal(line.project(p2), p2)

    other_point_on_line = np.array([0.0, 0.0, 2.0])
    np.testing.assert_array_almost_equal(
        line.project(other_point_on_line), other_point_on_line
    )

    example_perpendicular_displacement = [
        k * vg.perpendicular(line.along, vg.basis.x) for k in [0.1, 0.5, -2.0]
    ]
    for p in [p1, p2, other_point_on_line]:
        for displacement in example_perpendicular_displacement:
            np.testing.assert_array_almost_equal(line.project(p + displacement), p)


def test_project_stacked():
    p1 = np.array([5.0, 5.0, 4.0])
    p2 = np.array([10.0, 10.0, 6.0])
    line = Line.from_points(p1, p2)

    other_point_on_line = np.array([0.0, 0.0, 2.0])

    example_perpendicular_displacement = [
        k * vg.perpendicular(line.along, vg.basis.x) for k in [0.1, 0.5, -2.0]
    ]

    example_points = np.vstack([p1, p2, other_point_on_line])
    expected_projected_points = np.vstack([p1, p2, other_point_on_line])

    np.testing.assert_array_almost_equal(
        line.project(example_points), expected_projected_points
    )
    np.testing.assert_array_almost_equal(
        line.project(example_points + example_perpendicular_displacement),
        expected_projected_points,
    )
