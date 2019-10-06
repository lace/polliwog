import vg
from .._common.shape import check_shape_any


def project_to_line(points, reference_points_of_lines, vectors_along_lines):
    check_shape_any(points, (3,), (-1, 3), name="points")
    check_shape_any(
        reference_points_of_lines, (3,), (-1, 3), name="reference_points_of_lines"
    )
    if points.ndim == 2 and reference_points_of_lines.ndim == 2:
        vg.shape.check(locals(), "reference_points_of_lines", (len(points), 3))
    vg.shape.check(locals(), "vectors_along_lines", reference_points_of_lines.shape)

    return reference_points_of_lines + vg.project(
        points - reference_points_of_lines, onto=vectors_along_lines
    )
