import math
import numpy as np
import vg
import pytest
from .functions import (
    plane_equation_from_points,
    plane_normal_from_points,
    normal_and_offset_from_plane_equations,
    signed_distance_to_plane,
    project_point_to_plane,
)
from .plane import Plane
from .coordinate_planes import coordinate_planes


def assert_plane_equation_satisfies_points(plane_equation, points):
    a, b, c, d = plane_equation
    plane_equation_test = [a * x + b * y + c * z + d for x, y, z in points]
    assert np.any(plane_equation_test) == False


def test_plane_equation_from_points():
    points = np.array([[1, 1, 1], [-1, 1, 0], [2, 0, 3]])
    equation = plane_equation_from_points(points)
    assert_plane_equation_satisfies_points(equation, points)


def test_plane_equation_from_points_is_in_expected_orientation():
    equation = plane_equation_from_points(
        np.array([vg.basis.x, vg.basis.y, vg.basis.neg_x])
    )
    normal = equation[:3]
    np.testing.assert_array_equal(normal, vg.basis.z)


def test_plane_equation_from_points_stacked():
    points = np.array(
        [[[1, 1, 1], [-1, 1, 0], [2, 0, 3]], [vg.basis.x, vg.basis.y, vg.basis.neg_x]]
    )
    equations = plane_equation_from_points(points)
    for these_points, this_equation in zip(points, equations):
        assert_plane_equation_satisfies_points(this_equation, these_points)


def test_plane_normal_from_points():
    np.testing.assert_array_almost_equal(
        plane_normal_from_points(np.array([vg.basis.x, vg.basis.y, vg.basis.neg_x])),
        vg.basis.z,
    )


def test_plane_normal_from_points_stacked():
    np.testing.assert_array_almost_equal(
        plane_normal_from_points(
            np.array(
                [
                    [vg.basis.x, vg.basis.y, vg.basis.z],
                    [vg.basis.x, vg.basis.y, vg.basis.neg_x],
                ]
            )
        ),
        np.array([np.repeat(math.sqrt(1.0 / 3.0), 3), vg.basis.z]),
    )


def test_normal_and_offset_from_plane_equations():
    equations = plane_equation_from_points(
        np.array(
            [
                [vg.basis.x, vg.basis.y, vg.basis.z],
                [vg.basis.x, vg.basis.y, vg.basis.neg_x],
            ]
        )
    )
    normals, offsets = normal_and_offset_from_plane_equations(equations)
    np.testing.assert_array_almost_equal(
        normals, np.array([np.repeat(math.sqrt(1.0 / 3.0), 3), vg.basis.z])
    )


def test_signed_distances_for_xz_plane_at_origin():
    np.testing.assert_array_equal(
        signed_distance_to_plane(
            points=np.array([[500.0, 502.0, 503.0], [-500.0, -501.0, -503.0]]),
            plane_equations=coordinate_planes.xz.equation,
        ),
        np.array([502.0, -501.0]),
    )


def test_signed_distances_for_diagonal_plane():
    np.testing.assert_array_almost_equal(
        signed_distance_to_plane(
            points=np.array([[425.0, 425.0, 25.0], [-500.0, -500.0, 25.0]]),
            # Diagonal plane @ origin - draw a picture!
            plane_equations=Plane(
                point_on_plane=np.array([1.0, 1.0, 0.0]),
                unit_normal=vg.normalize(np.array([1.0, 1.0, 0.0])),
            ).equation,
        ),
        np.array(
            [math.sqrt(2 * (425.0 - 1.0) ** 2), -math.sqrt(2 * (500.0 + 1.0) ** 2)]
        ),
    )


def test_signed_distance_validation():
    with pytest.raises(
        ValueError,
        match=r"points must be an array with shape \(3,\) or \(-1, 3\); got \(1, 1, 1\)",
    ):
        signed_distance_to_plane(
            points=np.array([[[1.0]]]), plane_equations=coordinate_planes.xz.equation
        ),


def test_project_point_to_plane():
    np.testing.assert_array_equal(
        project_point_to_plane(
            points=np.array([10, 20, -5]),
            plane_equations=Plane(
                point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y
            ).equation,
        ),
        np.array([10, 10, -5]),
    )


def test_project_point_to_plane_vectorized_points():
    np.testing.assert_array_equal(
        project_point_to_plane(
            points=np.array([[10, 20, -5], [2, 7, 203]]),
            plane_equations=Plane(
                point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y
            ).equation,
        ),
        np.array([[10, 10, -5], [2, 10, 203]]),
    )


def test_project_point_to_plane_vectorized_planes():
    np.testing.assert_array_equal(
        project_point_to_plane(
            points=np.array([10, 20, -5]),
            plane_equations=np.array(
                [
                    Plane(
                        point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y
                    ).equation,
                    Plane(
                        point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y
                    ).equation,
                ]
            ),
        ),
        np.array([[10, 10, -5], [10, 10, -5]]),
    )


def test_project_point_to_plane_vectorized_both():
    np.testing.assert_array_equal(
        project_point_to_plane(
            points=np.array([[10, 20, -5], [10, 30, -5]]),
            plane_equations=np.array(
                [
                    Plane(
                        point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y
                    ).equation,
                    Plane(
                        point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y
                    ).equation,
                ]
            ),
        ),
        np.array([[10, 10, -5], [10, 10, -5]]),
    )


def test_project_point_to_plane_validation():
    with pytest.raises(
        ValueError,
        match=r"points must be an array with shape \(3,\) or \(-1, 3\); got \(1, 1, 1\)",
    ):
        project_point_to_plane(
            points=np.array([[[1.0]]]),
            plane_equations=Plane(
                point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y
            ).equation,
        )

    with pytest.raises(
        ValueError,
        match=r"plane_equations must be an array with shape \(3, 4\); got \(2, 4\)",
    ):
        project_point_to_plane(
            points=np.array([vg.basis.x, vg.basis.x, vg.basis.x]),
            plane_equations=np.array(
                [
                    Plane(
                        point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y
                    ).equation,
                    Plane(
                        point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y
                    ).equation,
                ]
            ),
        )
