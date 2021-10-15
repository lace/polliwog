import numpy as np
from vg.compat import v2 as vg
from ..tri import FACE_DTYPE


def slice_triangles_by_plane(
    vertices,
    faces,
    point_on_plane,
    plane_normal,
    faces_to_slice=None,
    ret_face_mapping=False,
):
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
    if faces_to_slice is not None:
        vg.shape.check(locals(), "faces_to_slice", (-1,))
        assert faces_to_slice.dtype == np.bool

    result = slice_faces_plane(
        vertices=vertices,
        faces=faces,
        plane_normal=plane_normal,
        plane_origin=point_on_plane,
        face_index=None if faces_to_slice is None else faces_to_slice.nonzero()[0],
        return_face_mapping=ret_face_mapping,
    )

    if ret_face_mapping:
        (
            vertices,
            faces,
            face_mapping,
        ) = result  # lgtm [py/mismatched-multiple-assignment]
        assert face_mapping.dtype == FACE_DTYPE
    else:
        vertices, faces = result
    assert vertices.dtype == np.float64
    assert faces.dtype == FACE_DTYPE

    return result
