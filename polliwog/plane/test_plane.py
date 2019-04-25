import math
import numpy as np
import vg
from .plane import Plane


def test_returns_signed_distances_for_xz_plane_at_origin():
    # x-z plane
    normal = np.array([0.0, 1.0, 0.0])
    sample = np.array([0.0, 0.0, 0.0])

    plane = Plane(sample, normal)

    pts = np.array([[500.0, 502.0, 503.0], [-500.0, -501.0, -503.0]])

    expected = np.array([502.0, -501.0])

    np.testing.assert_array_equal(expected, plane.signed_distance(pts))
    np.testing.assert_array_equal(expected[0], plane.signed_distance(pts[0]))


def test_returns_unsigned_distances_for_xz_plane_at_origin():
    # x-z plane
    normal = np.array([0.0, 1.0, 0.0])
    sample = np.array([0.0, 0.0, 0.0])

    plane = Plane(sample, normal)

    pts = np.array([[500.0, 502.0, 503.0], [-500.0, -501.0, -503.0]])

    expected = np.array([502.0, 501.0])

    np.testing.assert_array_equal(expected, plane.distance(pts))


def test_returns_signed_distances_for_diagonal_plane():
    # diagonal plane @ origin - draw a picture!
    normal = np.array([1.0, 1.0, 0.0])
    normal /= np.linalg.norm(normal)
    sample = np.array([1.0, 1.0, 0.0])

    plane = Plane(sample, normal)

    pts = np.array([[425.0, 425.0, 25.0], [-500.0, -500.0, 25.0]])

    expected = np.array(
        [math.sqrt(2 * (425.0 - 1.0) ** 2), -math.sqrt(2 * (500.0 + 1.0) ** 2)]
    )

    np.testing.assert_array_almost_equal(expected, plane.signed_distance(pts))


def test_returns_unsigned_distances_for_diagonal_plane_at_origin():
    # diagonal plane @ origin - draw a picture!
    normal = np.array([1.0, 1.0, 0.0])
    normal /= np.linalg.norm(normal)

    sample = np.array([0.0, 0.0, 0.0])

    plane = Plane(sample, normal)

    pts = np.array([[425.0, 425.0, 25.0], [-500.0, -500.0, 25.0]])

    expected = np.array([math.sqrt(2 * (425.0 ** 2)), math.sqrt(2 * (500.0 ** 2))])

    np.testing.assert_array_almost_equal(expected, plane.distance(pts))


def test_returns_sign_for_diagonal_plane():
    # diagonal plane @ origin - draw a picture!
    normal = np.array([1.0, 1.0, 0.0])
    normal /= np.linalg.norm(normal)
    sample = np.array([1.0, 1.0, 0.0])

    plane = Plane(sample, normal)

    pts = np.array([[425.0, 425.0, 25.0], [-500.0, -500.0, 25.0]])

    sign = plane.sign(pts)

    expected = np.array([1.0, -1.0])
    np.testing.assert_array_equal(sign, expected)


def test_canonical_point():
    normal = np.array([1.0, 1.0, 0.0])
    normal /= np.linalg.norm(normal)

    sample = np.array([0.0, 0.0, 0.0])

    plane = Plane(sample, normal)

    np.testing.assert_array_equal(plane.canonical_point, np.array([0.0, 0.0, 0.0]))

    plane = Plane(sample, -normal)

    np.testing.assert_array_equal(plane.canonical_point, np.array([0.0, 0.0, 0.0]))

    normal = np.array([1.0, 7.0, 9.0])
    normal /= np.linalg.norm(normal)

    plane = Plane(sample, normal)

    np.testing.assert_array_equal(plane.canonical_point, np.array([0.0, 0.0, 0.0]))

    plane = Plane(sample, -normal)

    np.testing.assert_array_equal(plane.canonical_point, np.array([0.0, 0.0, 0.0]))

    normal = np.array([1.0, 0.0, 0.0])
    normal /= np.linalg.norm(normal)

    sample = np.array([3.0, 10.0, 20.0])

    plane = Plane(sample, normal)

    np.testing.assert_array_equal(plane.canonical_point, np.array([3, 0.0, 0.0]))

    plane = Plane(sample, -normal)

    np.testing.assert_array_equal(plane.canonical_point, np.array([3, 0.0, 0.0]))

    normal = np.array([1.0, 1.0, 1.0])
    normal /= np.linalg.norm(normal)

    sample = np.array([1.0, 2.0, 10.0])

    plane = Plane(sample, normal)

    np.testing.assert_array_almost_equal(
        plane.canonical_point, np.array([4.333333, 4.333333, 4.333333])
    )

    plane = Plane(sample, -normal)

    np.testing.assert_array_almost_equal(
        plane.canonical_point, np.array([4.333333, 4.333333, 4.333333])
    )


def test_project_point():
    plane = Plane(point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y)

    point = np.array([10, 20, -5])

    expected = np.array([10, 10, -5])

    np.testing.assert_array_equal(plane.project_point(point), expected)


def test_project_point_vectorized():
    plane = Plane(point_on_plane=np.array([0, 10, 0]), unit_normal=vg.basis.y)

    points = np.array([[10, 20, -5], [2, 7, 203]])

    expected = np.array([[10, 10, -5], [2, 10, 203]])

    np.testing.assert_array_equal(plane.project_point(points), expected)


def test_plane_from_points():
    points = np.array([[1, 1, 1], [-1, 1, 0], [2, 0, 3]])
    plane = Plane.from_points(*points)

    a, b, c, d = plane.equation

    plane_equation_test = [a * x + b * y + c * z + d for x, y, z in points]
    np.testing.assert_array_equal(plane_equation_test, [0, 0, 0])

    projected_points = [plane.project_point(p) for p in points]
    np.testing.assert_array_almost_equal(projected_points, points)


def test_plane_from_points_order():
    points = np.array([[1, 0, 0], [0, math.sqrt(1.25), 0], [-1, 0, 0]])
    plane = Plane.from_points(*points)

    expected_v = np.array([0, 0, 1])
    np.testing.assert_array_equal(plane.normal, expected_v)


def test_plane_from_points_and_vector():
    p1 = np.array([1, 5, 7])
    p2 = np.array([-2, -2, -2])
    v = np.array([1, 0, -1])
    plane = Plane.from_points_and_vector(p1, p2, v)

    points = [p1, p2]
    projected_points = [plane.project_point(p) for p in points]
    np.testing.assert_array_almost_equal(projected_points, points)

    assert np.dot(v, plane.normal) == 0


def test_fit_from_points():
    # Set up a collection of points in the X-Y plane.
    np.random.seed(0)
    points = np.hstack([np.random.random((100, 2)), np.zeros(100).reshape(-1, 1)])
    plane = Plane.fit_from_points(points)

    # The normal vector should be closely aligned with the Z-axis.
    z_axis = np.array([0.0, 0.0, 1.0])
    angle = np.arccos(np.dot(plane.normal, z_axis) / np.linalg.norm(plane.normal))
    assert angle % np.pi < 1e-6


def test_line_plane_intersection():
    # x-z plane
    normal = np.array([0.0, 1.0, 0.0])
    sample = np.array([0.0, 0.0, 0.0])

    plane = Plane(sample, normal)
    # non-intersecting
    assert plane.line_xsection(pt=vg.basis.neg_y, ray=vg.basis.x) is None
    # coplanar
    assert plane.line_xsection(pt=np.zeros(3), ray=vg.basis.x) is None
    np.testing.assert_array_equal(
        plane.line_xsection(pt=vg.basis.neg_y, ray=vg.basis.y), np.zeros(3)
    )
    np.testing.assert_array_equal(
        plane.line_xsection(pt=vg.basis.neg_y, ray=np.array([1.0, 1.0, 0.0])),
        vg.basis.x,
    )


def test_line_plane_intersections():
    # x-z plane
    normal = np.array([0.0, 1.0, 0.0])
    sample = np.array([0.0, 0.0, 0.0])

    plane = Plane(sample, normal)
    pts = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    rays = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    expected = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    intersections, is_intersecting = plane.line_xsections(pts, rays)
    np.testing.assert_array_equal(intersections, expected)
    np.testing.assert_array_equal(is_intersecting, [False, False, True, True])


def test_line_segment_plane_intersection():
    # x-z plane
    normal = np.array([0.0, 1.0, 0.0])
    sample = np.array([0.0, 0.0, 0.0])

    plane = Plane(sample, normal)
    # non-intersecting
    assert (
        plane.line_segment_xsection(vg.basis.neg_y, np.array([1.0, -1.0, 0.0])) is None
    )
    # coplanar
    assert plane.line_segment_xsection(np.zeros(3), vg.basis.x) is None
    np.testing.assert_array_equal(
        plane.line_segment_xsection(vg.basis.neg_y, vg.basis.y), np.zeros(3)
    )
    np.testing.assert_array_equal(
        plane.line_segment_xsection(vg.basis.neg_y, np.array([2.0, 1.0, 0.0])),
        vg.basis.x,
    )
    # line intersecting, but not in segment
    assert plane.line_segment_xsection(vg.basis.y, np.array([0.0, 2.0, 0.0])) is None


def test_line_segment_plane_intersections():
    # x-z plane
    normal = np.array([0.0, 1.0, 0.0])
    sample = np.array([0.0, 0.0, 0.0])

    plane = Plane(sample, normal)
    a = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    b = np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    expected = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
        ]
    )
    intersections, is_intersecting = plane.line_segment_xsections(a, b)
    np.testing.assert_array_equal(intersections, expected)
    np.testing.assert_array_equal(is_intersecting, [False, False, True, True, False])
