import numpy as np
import vg
from ..plane.plane import Plane
from .cut_by_plane import cut_open_polyline_by_plane

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
    return (
        random_points_on_plane + signs.reshape(-1, 1) * random_displacement_along_normal
    )


def intersect_segment_with_plane(p1, p2):
    from ..plane.intersections import (
        intersect_segment_with_plane as _intersect_segment_with_plane,
    )

    return _intersect_segment_with_plane(
        start_points=np.array([p1]),
        segment_vectors=np.array([p2 - p1]),
        points_on_plane=np.array([point_on_plane]),
        plane_normals=np.array([plane_normal]),
    )[0]


def test_open_starts_in_front_ends_in_back():
    signs = np.array([1, 1, 1, 1, 1, -1, -1])
    vertices = vertices_with_signs(signs)
    result = cut_open_polyline_by_plane(vertices, plane)
    np.testing.assert_array_equal(
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
    pass


def test_open_starts_in_back_ends_in_front():
    signs = np.array([-1, -1, 1])
    pass


def test_open_starts_in_back_ends_in_front_with_vertex_on_plane():
    signs = np.array([-1, -1, -1, -1, -1, 0, 1, 1, 1])
    pass


def test_open_starts_on_plane():
    signs = np.array([0, 1, 1, 1, 1])
    pass


def test_open_starts_on_plane_then_in_back():
    signs = np.array([0, -1, -1, -1, -1])
    pass


def test_open_ends_on_plane():
    signs = np.array([1, 1, 1, 1, 0])
    pass


def test_open_in_back_then_ends_on_plane():
    signs = np.array([-1, -1, -1, -1, 0])
    pass


def test_open_starts_with_edges_in_plane():
    signs = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    pass


def test_open_starts_in_front_then_along_plane_then_in_front_again():
    signs = np.array([1, 1, 0, 0, 1])
    pass


def test_open_starts_in_front_then_along_plane_then_in_back():
    signs = np.array([1, 1, 0, 0, -1])
    pass


def test_open_all_in_front():
    signs = np.array([1, 1, 1])
    pass


def test_open_all_in_back():
    signs = np.array([-1, -1, -1])
    pass


def test_open_all_in_plane():
    signs = np.array([0, 0])
    pass


def test_one_vert_in_front():
    signs = np.array([1])
    pass


def test_one_vert_in_back():
    signs = np.array([-1])
    pass


def test_one_vert_on_plane():
    signs = np.array([0])
    pass
