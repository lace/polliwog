import vg


def coplanar_points_are_on_same_side_of_line(a, b, p1, p2, atol=1e-8):
    """
    Using "same-side technique" from http://blackpawn.com/texts/pointinpoly/default.html
    """
    along_line = b - a
    return vg.dot(vg.cross(along_line, p1 - a), vg.cross(along_line, p2 - a)) >= -atol


def contains_coplanar_point(a, b, c, point, atol=1e-8):
    """
    Assuming `point` is coplanar with the triangle `ABC`, check if it lies
    inside it.

    Using "same-side technique" from http://blackpawn.com/texts/pointinpoly/default.html
    """
    return (
        coplanar_points_are_on_same_side_of_line(b, c, point, a, atol=atol)
        and coplanar_points_are_on_same_side_of_line(a, c, point, b, atol=atol)
        and coplanar_points_are_on_same_side_of_line(a, b, point, c, atol=atol)
    )
