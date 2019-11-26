import numpy as np
from .quad_faces import quads_to_tris


def test_quads_to_tris():
    tris = np.array(
        [
            [3, 2, 1, 0],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
        ]
    )
    expected_quads = np.array(
        [
            [3, 2, 1],
            [3, 1, 0],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
        ]
    )
    expected_f_old_to_new = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    np.testing.assert_array_equal(quads_to_tris(tris), expected_quads)

    quads, f_old_to_new = quads_to_tris(tris, ret_mapping=True)
    np.testing.assert_array_equal(expected_quads, quads)
    np.testing.assert_array_equal(f_old_to_new, expected_f_old_to_new)
