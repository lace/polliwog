import numpy as np
from vg.compat import v2 as vg
from .functions import FACE_DTYPE


def quads_to_tris(quads, ret_mapping=False):
    """
    Convert quad faces to triangular faces.

    quads: An nx4 array.
    ret_mapping: A bool.

    When `ret_mapping` is `True`, return a 2nx3 array of new triangles and a 2nx3
    array mapping old quad indices to new trangle indices.

    When `ret_mapping` is `False`, return the 2nx3 array of triangles.
    """
    vg.shape.check(locals(), "quads", (-1, 4))

    tris = np.empty((2 * len(quads), 3), dtype=FACE_DTYPE)
    tris[0::2, :] = quads[:, [0, 1, 2]]
    tris[1::2, :] = quads[:, [0, 2, 3]]
    if ret_mapping:
        f_old_to_new = np.arange(len(tris), dtype=FACE_DTYPE).reshape(-1, 2)
        return tris, f_old_to_new
    else:
        return tris
