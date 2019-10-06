import numpy as np
import vg
from .functions import project_to_line


def test_project_to_line():
    p1 = np.array([5.0, 5.0, 4.0])
    p2 = np.array([10.0, 10.0, 6.0])
    along_line = p2 - p1

    common_kwargs = dict(reference_points_of_lines=p1, vectors_along_lines=along_line)

    np.testing.assert_array_almost_equal(
        project_to_line(points=p1, **common_kwargs), p1
    )
    np.testing.assert_array_almost_equal(
        project_to_line(points=p2, **common_kwargs), p2
    )

    other_point_on_line = np.array([0.0, 0.0, 2.0])
    np.testing.assert_array_almost_equal(
        project_to_line(points=other_point_on_line, **common_kwargs),
        other_point_on_line,
    )

    example_perpendicular_displacement = [
        k * vg.perpendicular(vg.normalize(along_line), vg.basis.x)
        for k in [0.1, 0.5, -2.0]
    ]
    for point_on_line in [p1, p2, other_point_on_line]:
        for displacement in example_perpendicular_displacement:
            np.testing.assert_array_almost_equal(
                project_to_line(points=point_on_line + displacement, **common_kwargs),
                point_on_line,
            )


def test_project_to_line_stacked_points():
    p1 = np.array([5.0, 5.0, 4.0])
    p2 = np.array([10.0, 10.0, 6.0])
    along_line = p2 - p1

    common_kwargs = dict(reference_points_of_lines=p1, vectors_along_lines=along_line)

    other_point_on_line = np.array([0.0, 0.0, 2.0])

    example_perpendicular_displacement = [
        k * vg.perpendicular(vg.normalize(along_line), vg.basis.x)
        for k in [0.1, 0.5, -2.0]
    ]

    example_points = np.vstack([p1, p2, other_point_on_line])
    expected_projected_points = np.vstack([p1, p2, other_point_on_line])

    np.testing.assert_array_almost_equal(
        project_to_line(points=example_points, **common_kwargs),
        expected_projected_points,
    )
    np.testing.assert_array_almost_equal(
        project_to_line(
            points=example_points + example_perpendicular_displacement, **common_kwargs
        ),
        expected_projected_points,
    )


def test_project_to_line_stacked_lines():
    p1 = np.array([5.0, 5.0, 4.0])
    p2 = np.array([10.0, 10.0, 6.0])
    along_line = p2 - p1

    common_kwargs = dict(
        reference_points_of_lines=np.array([p1, p1]),
        vectors_along_lines=np.array([along_line, along_line]),
    )

    other_point_on_line = np.array([0.0, 0.0, 2.0])
    np.testing.assert_array_almost_equal(
        project_to_line(points=other_point_on_line, **common_kwargs),
        np.array([other_point_on_line, other_point_on_line]),
    )

    example_perpendicular_displacement = [
        k * vg.perpendicular(vg.normalize(along_line), vg.basis.x)
        for k in [0.1, 0.5, -2.0]
    ]
    for point_on_line in [p1, p2, other_point_on_line]:
        for displacement in example_perpendicular_displacement:
            np.testing.assert_array_almost_equal(
                project_to_line(points=point_on_line + displacement, **common_kwargs),
                np.array([point_on_line, point_on_line]),
            )


def test_project_to_line_stacked_both():
    p1 = np.array([5.0, 5.0, 4.0])
    p2 = np.array([10.0, 10.0, 6.0])
    along_line = p2 - p1

    common_kwargs = dict(
        reference_points_of_lines=np.array([p1, p1, p1]),
        vectors_along_lines=np.array([along_line, along_line, along_line]),
    )

    other_point_on_line = np.array([0.0, 0.0, 2.0])

    example_perpendicular_displacement = [
        k * vg.perpendicular(vg.normalize(along_line), vg.basis.x)
        for k in [0.1, 0.5, -2.0]
    ]

    example_points = np.vstack([p1, p2, other_point_on_line])
    expected_projected_points = np.vstack([p1, p2, other_point_on_line])

    np.testing.assert_array_almost_equal(
        project_to_line(points=example_points, **common_kwargs),
        expected_projected_points,
    )
    np.testing.assert_array_almost_equal(
        project_to_line(
            points=example_points + example_perpendicular_displacement, **common_kwargs
        ),
        expected_projected_points,
    )
