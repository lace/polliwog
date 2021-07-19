from vg.compat import v2 as vg
from .._common.shape import check_shape_any

__all__ = ["project_point_to_line", "coplanar_points_are_on_same_side_of_line"]


def project_point_to_line(points, reference_points_of_lines, vectors_along_lines):
    """
    Project a point to a line, or pairwise project a stack of points to a
    stack of lines.
    """
    k = check_shape_any(points, (3,), (-1, 3), name="points")
    check_shape_any(
        reference_points_of_lines,
        (3,),
        (-1 if k is None else k, 3),
        name="reference_points_of_lines",
    )
    vg.shape.check(locals(), "vectors_along_lines", reference_points_of_lines.shape)

    return reference_points_of_lines + vg.project(
        points - reference_points_of_lines, onto=vectors_along_lines
    )


def coplanar_points_are_on_same_side_of_line(a, b, p1, p2):
    """
    Test if the given points are on the same side of the given line.

    Args:
        a (np.arraylike): The first 3D point of interest.
        b (np.arraylike): The second 3D point of interest.
        p1 (np.arraylike): A first point which lies on the line of interest.
        p2 (np.arraylike): A second point which lies on the line of interest.

    Returns:
        bool: `True` when `a` and `b` are on the same side of the line defined
        by `p1` and `p2`.
    """
    check_shape_any(a, (3,), (-1, 3), name="a")
    vg.shape.check(locals(), "b", a.shape)
    vg.shape.check(locals(), "p1", a.shape)
    vg.shape.check(locals(), "p2", a.shape)

    # Uses "same-side technique" from http://blackpawn.com/texts/pointinpoly/default.html
    along_line = b - a
    return vg.dot(vg.cross(along_line, p1 - a), vg.cross(along_line, p2 - a)) >= 0
