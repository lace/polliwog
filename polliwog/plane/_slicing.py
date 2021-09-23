import numpy as np
from vg.compat import v2 as vg


def slice_triangles_by_plane(vertices, faces, point_on_plane, plane_normal):
    """
    Slice the given triangles by the given plane.

    - Triangles partially in front of the plane are sliced.
    - Triangles fully in front of the plane are kept as is.
    - Triangles fully behind the plane are culled.

    Returns a tuple: vertices, faces.
    """
    from ._trimesh_intersections import slice_faces_plane

    vg.shape.check(locals(), "vertices", (-1, 3))
    vg.shape.check(locals(), "faces", (-1, 3))
    vg.shape.check(locals(), "point_on_plane", (3,))
    vg.shape.check(locals(), "plane_normal", (3,))

    vertices, faces = slice_faces_plane(
        vertices=vertices,
        faces=faces,
        plane_normal=plane_normal,
        plane_origin=point_on_plane,
    )
    assert vertices.dtype == np.float64
    assert faces.dtype == np.uint64
    return vertices, faces
