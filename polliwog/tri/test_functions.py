import math
import numpy as np
from polliwog.tri import (
    barycentric_coordinates_of_points,
    edges_of_faces,
    sample,
    surface_area,
    surface_normals,
    tri_contains_coplanar_point,
)
import pytest
from vg.compat import v2 as vg


def test_edges_of_faces():
    faces = np.arange(6).reshape(-1, 3)
    np.testing.assert_array_equal(
        edges_of_faces(faces, normalize=False),
        np.array(
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [3, 4],
                [4, 5],
                [5, 3],
            ]
        ),
    )
    np.testing.assert_array_equal(
        edges_of_faces(faces),
        np.array(
            [
                [0, 1],
                [1, 2],
                [0, 2],
                [3, 4],
                [4, 5],
                [3, 5],
            ]
        ),
    )


def test_surface_normals_from_points_single():
    points = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

    np.testing.assert_allclose(
        surface_normals(points), np.array([3 ** -0.5, 3 ** -0.5, 3 ** -0.5])
    )

    np.testing.assert_allclose(
        surface_normals(points, normalize=False), np.array([9.0, 9.0, 9.0])
    )


def test_surface_normals_from_points_vectorized():
    from polliwog.shapes import triangular_prism

    p1 = np.array([3.0, 0.0, 0.0])
    p2 = np.array([0.0, 3.0, 0.0])
    p3 = np.array([0.0, 0.0, 3.0])
    vertices = triangular_prism(p1, p2, p3, 1.0)

    expected_normals = vg.normalize(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -2.0],
                [1.0, 1.0, -2.0],
                [-2.0, 1.0, 1.0],
                [-2.0, 1.0, 1.0],
                [1.0, -2.0, 1.0],
                [1.0, -2.0, 1.0],
                [-1.0, -1.0, -1.0],
            ]
        )
    )

    np.testing.assert_allclose(surface_normals(vertices), expected_normals)


def test_surface_area_single():
    # Example taken from https://math.stackexchange.com/q/897071/640314
    np.testing.assert_almost_equal(
        surface_area(np.array([[-1, 0, 2], [2, -1, 3], [4, 0, 1]])),
        1.5 * math.sqrt(10),
    )


def test_surface_area_vectorized():
    np.testing.assert_array_almost_equal(
        surface_area(
            np.array(
                [
                    [[-1, 0, 2], [2, -1, 3], [4, 0, 1]],
                    [[0, 0, 0], [0, 1, 0], [1, 1, 0]],
                ]
            )
        ),
        np.array([1.5 * math.sqrt(10), 0.5]),
    )


def test_tri_contains_coplanar_point():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([4.0, 0.1, 0.0])
    c = np.array([3.0, 3.1, 0.0])

    # Not sure why, but `is True` does not work.
    assert tri_contains_coplanar_point(a, b, c, a) == True  # noqa: E712
    assert tri_contains_coplanar_point(a, b, c, b) == True  # noqa: E712
    assert tri_contains_coplanar_point(a, b, c, c) == True  # noqa: E712
    assert (
        tri_contains_coplanar_point(a, b, c, np.array([2.0, 1.0, 0.0]))
        == True  # noqa: E712
    )

    # Unexpected, as it's not in the plane, though if projected to the plane,
    # it is in the triangle.
    assert (
        tri_contains_coplanar_point(a, b, c, np.array([0.0, 0.0, 1.0]))
        == True  # noqa: E712
    )

    assert (
        tri_contains_coplanar_point(a, b, c, np.array([2.0, 0.0, 0.0]))
        == False  # noqa: E712
    )
    assert (
        tri_contains_coplanar_point(a, b, c, np.array([2.0, 5.0, 0.0]))
        == False  # noqa: E712
    )

    assert (
        tri_contains_coplanar_point(
            np.array([0.06710189, 1.69908346, 0.06590126]),
            np.array([0.05648619, 1.70207, 0.07402092]),
            np.array([0.05969098, 1.69641423, 0.07268801]),
            np.array([0.07534771, 1.6869296, 0.06190757]),
        )
        == False  # noqa: E712
    )


def test_tri_contains_coplanar_point_stacked():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([4.0, 0.1, 0.0])
    c = np.array([3.0, 3.1, 0.0])

    stacked_a = np.array([a, a, a, a, a, a, a, [0.06710189, 1.69908346, 0.06590126]])
    stacked_b = np.array([b, b, b, b, b, b, b, [0.05648619, 1.70207, 0.07402092]])
    stacked_c = np.array([c, c, c, c, c, c, c, [0.05969098, 1.69641423, 0.07268801]])
    stacked_p = np.array(
        [
            a,
            b,
            c,
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.07534771, 1.6869296, 0.06190757],
        ]
    )
    np.testing.assert_array_equal(
        tri_contains_coplanar_point(stacked_a, stacked_b, stacked_c, stacked_p),
        np.array([True, True, True, True, True, False, False, False]),
    )


def test_barycentric():
    triangle = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, math.sqrt(2), 0.0]])

    points_of_interest = np.array(
        [
            [0.0, math.sqrt(2) / 3.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, math.sqrt(2), 0.0],
        ]
    )
    expected = np.array(
        [np.full((3,), 1.0 / 3.0), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # Perform all of our queries on the same triangle.
    tiled_triangle = np.tile(triangle.reshape(-1), len(points_of_interest)).reshape(
        -1, 3, 3
    )

    np.testing.assert_array_almost_equal(
        barycentric_coordinates_of_points(tiled_triangle, points_of_interest), expected
    )


def test_sample_returns_expected_centroid():
    # The first tri is scaled by 3x, giivng it 9x the area.
    num_samples = 100000
    points = sample(
        vertices_of_tris=np.array(
            [
                [[0, 0, 0], [9, 0, 0], [9, 12, 0]],
                [[0, 0, 0], [3, 0, 0], [3, 4, 0]],
            ]
        ),
        num_samples=num_samples,
    )

    assert len(points) == num_samples

    # To compute the expected centroid, weight the centroids of each triangle by
    # the triangle's area.
    centroids_of_tris = np.array(
        [
            [6, 4, 0],
            [2, 4 / 3, 0],
        ]
    )
    expected_centroid_of_points = vg.average(
        centroids_of_tris, weights=np.array([9, 1])
    )
    np.testing.assert_array_almost_equal(
        vg.average(points), expected_centroid_of_points, decimal=2
    )


def test_sample_weights():
    # The first tri is scaled by 3x, giivng it 9x the area, but we override the weights.
    num_samples = 100000
    weights = np.array([1, 4])
    points = sample(
        vertices_of_tris=np.array(
            [
                [[0, 0, 0], [9, 0, 0], [9, 12, 0]],
                [[0, 0, 0], [3, 0, 0], [3, 4, 0]],
            ]
        ),
        num_samples=num_samples,
        weights=weights,
    )

    assert len(points) == num_samples

    # To compute the expected centroid, weight the centroids of each triangle by
    # the triangle's area.
    centroids_of_tris = np.array(
        [
            [6, 4, 0],
            [2, 4 / 3, 0],
        ]
    )
    expected_centroid_of_points = vg.average(centroids_of_tris, weights=weights)
    np.testing.assert_array_almost_equal(
        vg.average(points), expected_centroid_of_points, decimal=2
    )


def test_sample_returns_expected_face_indices():
    num_samples = 100000
    _, face_indices = sample(
        vertices_of_tris=np.array([[[0, 0, 0], [3, 0, 0], [3, 4, 0]]]),
        num_samples=num_samples,
        ret_face_indices=True,
    )
    np.testing.assert_array_equal(face_indices, np.zeros(num_samples))


def test_sample_is_deterministic():
    common_kwargs = dict(
        vertices_of_tris=np.array([[[0, 0, 0], [3, 0, 0], [3, 4, 0]]]),
        num_samples=100000,
    )
    np.testing.assert_array_equal(sample(**common_kwargs), sample(**common_kwargs))


def test_sample_empty():
    points = sample(
        vertices_of_tris=np.zeros((0, 3, 3)),
        num_samples=10000,
    )
    np.testing.assert_array_equal(points, np.zeros((0, 3)))

    _, face_indices = sample(
        vertices_of_tris=np.zeros((0, 3, 3)),
        num_samples=10000,
        ret_face_indices=True,
    )
    np.testing.assert_array_equal(face_indices, np.zeros((0,)))


def test_sample_errors():
    with pytest.raises(ValueError, match="Expected num_samples to be an int"):
        sample(
            vertices_of_tris=np.array([[[0, 0, 0], [3, 0, 0], [3, 4, 0]]]),
            num_samples="nope",
        )
    with pytest.raises(
        ValueError, match="Expected rng to be an instance of np.random.Generator"
    ):
        sample(
            vertices_of_tris=np.array([[[0, 0, 0], [3, 0, 0], [3, 4, 0]]]),
            num_samples=10000,
            rng="nope",
        )
