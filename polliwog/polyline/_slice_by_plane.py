import numpy as np
from vg.compat import v2 as vg


def slice_open_polyline_by_plane(vertices, plane):
    from .. import Plane
    from ..plane import intersect_segment_with_plane

    num_v = vg.shape.check(locals(), "vertices", (-1, 3))
    if num_v == 0:
        raise ValueError("A plane can't intersect a polyline with no points")
    if not isinstance(plane, Plane):
        raise ValueError("plane should be an instance of polliwog.Plane")

    signed_distances = plane.signed_distance(vertices)
    signs_of_vertices = np.sign(signed_distances)

    (transition_points,) = (signs_of_vertices[:-1] != signs_of_vertices[1:]).nonzero()
    components = np.vsplit(vertices, transition_points + 1)
    component_signs = signs_of_vertices[np.concatenate([[0], transition_points + 1])]

    (components_in_front,) = (component_signs == 1).nonzero()
    if len(components_in_front) == 0:
        raise ValueError("Polyline has no vertices in front of the plane")
    elif len(components_in_front) > 1:
        raise ValueError("Polyline intersects the plane too many times")
    elif len(components) < 2:
        raise ValueError("Polyline lies entirely in front of the plane")
    (component_in_front,) = components_in_front
    verts_in_front = components[component_in_front]

    # Prepend the plane intersection point with the previous component.
    if component_in_front > 0:
        adjacent_vertex = components[component_in_front - 1][-1]
        sign_of_adjacent_vertex = component_signs[component_in_front - 1]
        if sign_of_adjacent_vertex == 0:
            prepend = adjacent_vertex
        else:
            prepend = intersect_segment_with_plane(
                start_points=adjacent_vertex,
                segment_vectors=verts_in_front[0] - adjacent_vertex,
                points_on_plane=plane.reference_point,
                plane_normals=plane.normal,
            )
    else:
        prepend = np.zeros((0, 3))

    # Append the plane intersection point with the next component.
    if component_in_front + 1 < len(components):
        adjacent_vertex = components[component_in_front + 1][0]
        sign_of_adjacent_vertex = component_signs[component_in_front + 1]
        if sign_of_adjacent_vertex == 0:
            append = adjacent_vertex
        else:
            last_vert = verts_in_front[-1]
            append = intersect_segment_with_plane(
                start_points=last_vert,
                segment_vectors=adjacent_vertex - last_vert,
                points_on_plane=plane.reference_point,
                plane_normals=plane.normal,
            )
    else:
        append = np.zeros((0, 3))

    return np.vstack([prepend, verts_in_front, append])
