import vg
from .._common.shape import check_shape_any

__all__ = ["project_to_line"]


def project_to_line(points, reference_points_of_lines, vectors_along_lines):
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
