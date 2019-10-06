import numpy as np
import vg
from .._common.shape import columnize, check_shape_any


def plane_normal_from_points(points):
    """
    Given a set of three points, compute the normal of the plane which
    passes through them. Also works on stacked inputs (i.e. many sets
    of three points).
    """
    points, _, transform_result = columnize(points, (-1, 3, 3), name="points")

    p1s = points[:, 0]
    p2s = points[:, 1]
    p3s = points[:, 2]
    v1s = p2s - p1s
    v2s = p3s - p1s
    unit_normals = vg.normalize(vg.cross(v1s, v2s))

    return transform_result(unit_normals)


def plane_equation_from_points(points):
    """
    Given many sets of three points, return a stack of plane equations
    [`A`, `B`, `C`, `D`] which satisfy `Ax + By + Cz + D = 0`. Also
    works on three points to return a single plane equation.
    
    These coefficients can be decomposed into the plane normal vector
    which is `[A, B, C]` and the offset `D`, either by the caller or
    by using `normal_and_offset_from_plane_equations()`.
    """
    points, _, transform_result = columnize(points, (-1, 3, 3), name="points")

    p1s = points[:, 0]
    unit_normals = plane_normal_from_points(points)
    D = -vg.dot(p1s, unit_normals)

    return transform_result(np.hstack([unit_normals, D.reshape(-1, 1)]))


def normal_and_offset_from_plane_equations(plane_equations):
    """
    Given `A`, `B`, `C`, `D` of the plane equation `Ax + By + Cz + D = 0`,
    return the plane normal vector which is `[A, B, C]` and the offset `D`.
    """
    check_shape_any(plane_equations, (4,), (-1, 4), name="plane_equations")

    if plane_equations.ndim == 2:
        normal = plane_equations[:, :3]
        offset = plane_equations[:, 3]
    else:
        normal = plane_equations[:3]
        offset = plane_equations[3]
    return normal, offset


def signed_distance_to_plane(points, plane_equations):
    """
    Return the signed distances from each point to the corresponding plane.

    For convenience, can also be called with a single point and a single
    plane.
    """
    check_shape_any(points, (3,), (-1, 3), name="points")
    check_shape_any(plane_equations, (4,), (-1, 4), name="plane_equations")
    if points.ndim == 2 and plane_equations.ndim == 2:
        vg.shape.check(locals(), "plane_equations", (len(points), 4))

    normals, offsets = normal_and_offset_from_plane_equations(plane_equations)
    return vg.dot(points, normals) + offsets


def project_point_to_plane(points, plane_equations):
    """
    Project each point to the corresponding plane.
    """
    check_shape_any(points, (3,), (-1, 3), name="points")
    check_shape_any(plane_equations, (4,), (-1, 4), name="plane_equations")
    if points.ndim == 2 and plane_equations.ndim == 2:
        vg.shape.check(locals(), "plane_equations", (len(points), 4))

    # Translate the point back to the plane along the normal.
    signed_distance = signed_distance_to_plane(points, plane_equations)
    normals, _ = normal_and_offset_from_plane_equations(plane_equations)

    if np.isscalar(signed_distance):
        return points - signed_distance * normals
    else:
        return points - signed_distance.reshape(-1, 1) * normals
