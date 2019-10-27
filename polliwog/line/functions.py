import vg
from .._common.shape import check_shape_any


def project_to_line(points, reference_points_of_lines, vectors_along_lines):
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
