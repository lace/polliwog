import numbers
import numpy as np
from vg.compat import v2 as vg
from .._common.shape import check_shape_any, columnize

__all__ = [
    "plane_normal_from_points",
    "plane_equation_from_points",
    "normal_and_offset_from_plane_equations",
    "signed_distance_to_plane",
    "project_point_to_plane",
    "mirror_point_across_plane",
]


def plane_normal_from_points(points, normalize=True):
    """
    Given a set of three points, compute the normal of the plane which
    passes through them. Also works on stacked inputs (i.e. many sets
    of three points).

    This is the same as `polliwog.tri.functions.surface_normals`, to
    which this delegates.
    """
    from ..tri import surface_normals

    return surface_normals(points=points, normalize=normalize)


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
    k = check_shape_any(points, (3,), (-1, 3), name="points")
    check_shape_any(
        plane_equations, (4,), (-1 if k is None else k, 4), name="plane_equations"
    )

    normals, offsets = normal_and_offset_from_plane_equations(plane_equations)
    return vg.dot(points, normals) + offsets


def translate_points_along_plane_normal(points, plane_equations, factor):
    """
    Translate each point along the plane normal. The distance translated
    is specified in terms of its initial distance from the plane.

    Args:
        factor (float): Transform this multiple of the normal. With `0`,
            this returns the identity. With `-1`, this projects to the
            plane. When `-2` it returns the opposite point.
    """
    k = check_shape_any(points, (3,), (-1, 3), name="points")
    check_shape_any(
        plane_equations, (4,), (-1 if k is None else k, 4), name="plane_equations"
    )
    assert isinstance(factor, numbers.Real)

    # Translate the point back to the plane along the normal.
    signed_distance = signed_distance_to_plane(points, plane_equations)
    normals, _ = normal_and_offset_from_plane_equations(plane_equations)

    if np.isscalar(signed_distance):
        return points + factor * signed_distance * normals
    else:
        return points + factor * signed_distance.reshape(-1, 1) * normals


def project_point_to_plane(points, plane_equations):
    """
    Project each point to the corresponding plane.
    """
    k = check_shape_any(points, (3,), (-1, 3), name="points")
    check_shape_any(
        plane_equations, (4,), (-1 if k is None else k, 4), name="plane_equations"
    )

    return translate_points_along_plane_normal(
        points=points, plane_equations=plane_equations, factor=-1
    )


def mirror_point_across_plane(points, plane_equations):
    """
    Mirror each point to the corresponding point on the opposite side of the
    plane.
    """
    k = check_shape_any(points, (3,), (-1, 3), name="points")
    check_shape_any(
        plane_equations, (4,), (-1 if k is None else k, 4), name="plane_equations"
    )

    return translate_points_along_plane_normal(
        points=points, plane_equations=plane_equations, factor=-2
    )
