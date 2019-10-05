import numpy as np
import vg


def columnize(arr, shape=(-1, 3)):
    """
    Helper for functions which may accept many stacks of three points (kx3)
    returning a stack of results, or a single set of three points (3x1)
    returning a single result.

    Returns the points as kx3, and a `transform_result` function which can
    be applied to the result. It picks off the first result in the 3x1 case.

    Not limited to kx3; this can be used different dimensional shapes like
    kx4, or higher dimensional shapes like kx3x3.
    """
    if not isinstance(shape, tuple):
        raise ValueError("shape should be a tuple")

    if arr.ndim == len(shape):
        vg.shape.check(locals(), "arr", shape)
        return arr, True, lambda x: x
    else:
        vg.shape.check(locals(), "arr", shape[1:])
        return arr.reshape(*shape), False, lambda x: x[0]


def columnize_together(points, plane_equations):
    """
    If points and plane_equations are both stacked, they must be the same
    length. Enforce this constraint, then return columnized versions of
    both inputs.
    """
    if points.ndim == 2 and plane_equations.ndim == 2:
        # If points and plane_equations are both stacked, they must be the same
        # length.
        k = vg.shape.check(locals(), "points", (-1, 3))
        vg.shape.check(locals(), "plane_equations", (k, 4))
        return points, plane_equations, lambda x: x
    else:
        columnized_points, _, _ = columnize(points, (-1, 3))
        columnized_plane_equations, _, _ = columnize(plane_equations, (-1, 4))
        input_is_columnized = points.ndim == 2 or plane_equations.ndim == 2
        return columnized_points, columnized_plane_equations, lambda x: x[0]


def plane_normal_from_points(points):
    """
    Given a set of three points, compute the normal of the plane which
    passes through them. Also works on stacked inputs (i.e. many sets
    of three points).
    """
    points, _, transform_result = columnize(points, (-1, 3, 3))

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
    points, _, transform_result = columnize(points, (-1, 3, 3))

    p1s = points[:, 0]
    unit_normals = plane_normal_from_points(points)
    D = -vg.dot(p1s, unit_normals)

    return transform_result(np.hstack([unit_normals, D.reshape(-1, 1)]))


def normal_and_offset_from_plane_equations(plane_equations):
    """
    Given `A`, `B`, `C`, `D` of the plane equation `Ax + By + Cz + D = 0`,
    return the plane normal vector which is `[A, B, C]` and the offset `D`.
    """
    if plane_equations.ndim == 2:
        vg.shape.check(locals(), "plane_equations", (-1, 4))
        normal = plane_equations[:, :3]
        offset = plane_equations[:, 3]
    else:
        vg.shape.check(locals(), "plane_equations", (4,))
        normal = plane_equations[:3]
        offset = plane_equations[3]
    return normal, offset


def signed_distance_to_plane(points, plane_equations):
    """
    Return the signed distances from each point to the corresponding plane.

    For convenience, can also be called with a single point and a single
    plane.
    """
    input_is_columnized = points.ndim == 2 or plane_equations.ndim == 2
    normals, offsets = normal_and_offset_from_plane_equations(plane_equations)
    return vg.dot(points, normals) + offsets


def project_point_to_plane(points, plane_equations):
    """
    Project each point to the corresponding plane.
    """
    input_is_columnized = points.ndim == 2 or plane_equations.ndim == 2

    # Translate the point back to the plane along the normal.
    normals, _ = normal_and_offset_from_plane_equations(plane_equations)
    signed_distance = signed_distance_to_plane(points, plane_equations)
    if np.isscalar(signed_distance):
        return points - signed_distance * normals
    else:
        return points - signed_distance.reshape(-1, 1) * normals
