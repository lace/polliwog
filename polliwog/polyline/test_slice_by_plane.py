import numpy as np
import pytest
import vg
from ._slice_by_plane import slice_open_polyline_by_plane
from .. import Plane

point_on_plane = np.array([1.0, 2.0, 3.0])
plane_normal = vg.normalize(np.array([3.0, 4.0, 5.0]))
plane = Plane(point_on_plane=point_on_plane, unit_normal=plane_normal)


def rand_nonzero(*shape):
    return 128 * np.random.rand(*shape) + 1e-6


def vertices_with_signs(signs):
    num_verts = len(signs)
    random_points_on_plane = plane.project_point(rand_nonzero(num_verts, 3))
    random_displacement_along_normal = (
        rand_nonzero(num_verts).reshape(-1, 1) * plane_normal
    )
    vertices = (
        random_points_on_plane + signs.reshape(-1, 1) * random_displacement_along_normal
    )
    # Because of rounding, the random points don't necessarily return 0 for
    # sign, so pick one that does.
    vertices[signs == 0] = plane.reference_point
    np.testing.assert_array_equal(plane.sign(vertices), signs)
    return vertices


def intersect_segment_with_plane(p1, p2):
    from ..plane import intersect_segment_with_plane as _intersect_segment_with_plane

    return _intersect_segment_with_plane(
        start_points=p1,
        segment_vectors=p2 - p1,
        points_on_plane=point_on_plane,
        plane_normals=plane_normal,
    )


def test_open_starts_in_front_ends_in_back():
    signs = np.array([1, 1, 1, 1, 1, -1, -1])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_almost_equal(
        result,
        np.vstack(
            [
                vertices[signs == 1],
                [intersect_segment_with_plane(vertices[4], vertices[5])],
            ]
        ),
    )


def test_open_starts_in_front_ends_in_back_with_vertex_on_plane():
    signs = np.array([1, 1, 1, 0, -1, -1, -1, -1])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_equal(result, vertices[signs >= 0])


def test_open_starts_in_back_ends_in_front():
    signs = np.array([-1, -1, 1])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_almost_equal(
        result,
        np.vstack(
            [
                [intersect_segment_with_plane(vertices[1], vertices[2])],
                vertices[signs == 1],
            ]
        ),
    )


def test_open_starts_in_back_ends_in_front_with_vertex_on_plane():
    signs = np.array([-1, -1, -1, -1, -1, 0, 1, 1, 1])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_equal(result, vertices[signs >= 0])


def test_open_starts_on_plane_ends_in_front():
    signs = np.array([0, 1, 1, 1, 1])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_equal(result, vertices)


def test_open_starts_with_edges_along_plane_ends_in_front():
    signs = np.array([0, 0, 1, 1, 1, 1])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_equal(result, vertices[1:])


def test_open_starts_on_plane_ends_in_back():
    signs = np.array([0, -1, -1, -1, -1])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="Polyline has no vertices in front of the plane"
    ):
        slice_open_polyline_by_plane(vertices, plane)


def test_open_starts_in_front_ends_on_plane():
    signs = np.array([1, 1, 1, 1, 0])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_equal(result, vertices)


def test_open_starts_in_front_ends_with_edges_along_plane():
    signs = np.array([1, 1, 1, 1, 0, 0])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_equal(result, vertices[:-1])


def test_open_in_back_then_ends_on_plane():
    signs = np.array([-1, -1, -1, -1, 0])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="Polyline has no vertices in front of the plane"
    ):
        slice_open_polyline_by_plane(vertices, plane)


def test_open_starts_in_front_then_along_plane_then_in_front_again():
    signs = np.array([1, 1, 0, 0, 1])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="Polyline intersects the plane too many times"
    ):
        slice_open_polyline_by_plane(vertices, plane)


def test_open_starts_in_back_then_in_front_then_in_back_again():
    signs = np.array([-1, -1, 1, 1, 1, -1, -1])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_almost_equal(
        result,
        np.vstack(
            [
                [intersect_segment_with_plane(vertices[1], vertices[2])],
                vertices[signs == 1],
                [intersect_segment_with_plane(vertices[4], vertices[5])],
            ]
        ),
    )


def test_open_starts_in_back_then_along_plane_then_in_front_then_in_back_again():
    signs = np.array([-1, -1, 0, 0, 1, 1, 1, -1, -1])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_almost_equal(
        result,
        np.vstack(
            [
                vertices[3],
                vertices[signs == 1],
                [intersect_segment_with_plane(vertices[6], vertices[7])],
            ]
        ),
    )


def test_open_starts_in_front_then_along_plane_then_in_back():
    signs = np.array([1, 1, 0, 0, -1])
    vertices = vertices_with_signs(signs)

    result = slice_open_polyline_by_plane(vertices, plane)

    np.testing.assert_array_equal(result, vertices[0:3])


def test_open_all_in_front():
    signs = np.array([1, 1, 1])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="Polyline lies entirely in front of the plane"
    ):
        slice_open_polyline_by_plane(vertices, plane)


def test_open_all_in_back():
    signs = np.array([-1, -1, -1])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="Polyline has no vertices in front of the plane"
    ):
        slice_open_polyline_by_plane(vertices, plane)


def test_open_all_in_plane():
    signs = np.array([0, 0])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="Polyline has no vertices in front of the plane"
    ):
        slice_open_polyline_by_plane(vertices, plane)


def test_open_one_vert_in_front():
    signs = np.array([1])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="Polyline lies entirely in front of the plane"
    ):
        slice_open_polyline_by_plane(vertices, plane)


def test_open_one_vert_in_back():
    signs = np.array([-1])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="Polyline has no vertices in front of the plane"
    ):
        slice_open_polyline_by_plane(vertices, plane)


def test_open_one_vert_on_plane():
    signs = np.array([0])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="Polyline has no vertices in front of the plane"
    ):
        slice_open_polyline_by_plane(vertices, plane)


def test_open_not_a_plane():
    signs = np.array([1, -1])
    vertices = vertices_with_signs(signs)

    not_a_plane = []

    with pytest.raises(
        ValueError, match="plane should be an instance of polliwog.Plane"
    ):
        slice_open_polyline_by_plane(vertices, not_a_plane)


def test_open_no_points():
    signs = np.array([])
    vertices = vertices_with_signs(signs)

    with pytest.raises(
        ValueError, match="A plane can't intersect a polyline with no points"
    ):
        slice_open_polyline_by_plane(vertices, plane)
