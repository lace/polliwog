import numpy as np
from vg.compat import v2 as vg

__all__ = [
    "transform_matrix_for_non_uniform_scale",
    "transform_matrix_for_rotation",
    "transform_matrix_for_translation",
    "transform_matrix_for_uniform_scale",
]


def _convert_33_to_44(matrix):
    """
    Transform from:
        array([[1., 2., 3.],
               [2., 3., 4.],
               [5., 6., 7.]])
    to:
        array([[1., 2., 3., 0.],
               [2., 3., 4., 0.],
               [5., 6., 7., 0.],
               [0., 0., 0., 1.]])

    """
    vg.shape.check(locals(), "matrix", (3, 3))
    result = np.pad(matrix, ((0, 1), (0, 1)), mode="constant")
    result[3][3] = 1
    return result


def transform_matrix_for_rotation(rotation, ret_inverse_matrix=False):
    """
    Create a transformation matrix from the given 3x3 rotation matrix or a
    Rodrigues vector.

    With `ret_inverse_matrix=True`, also returns a matrix which provides
    the reverse transform.
    """
    from ._rodrigues import rodrigues_vector_to_rotation_matrix

    if rotation.shape == (3, 3):
        forward3 = rotation
    else:
        vg.shape.check(locals(), "rotation", (3,))
        forward3 = rodrigues_vector_to_rotation_matrix(rotation)

    forward = _convert_33_to_44(forward3)

    if not ret_inverse_matrix:
        return forward

    # The inverse of a rotation matrix is its transpose.
    inverse = forward.T
    return forward, inverse


def transform_matrix_for_translation(translation, ret_inverse_matrix=False):
    """
    Create a transformation matrix which translates by the provided
    displacement vector.

    Forward:

        [[  1,  0,  0,  v_0 ],
        [  0,  1,  0,  v_1 ],
        [  0,  0,  1,  v_2 ],
        [  0,  0,  0,  1   ]]

    Reverse:

        [[  1,  0,  0,  -v_0 ],
        [  0,  1,  0,  -v_1 ],
        [  0,  0,  1,  -v_2 ],
        [  0,  0,  0,  1    ]]

    Args:
        vector (np.arraylike): A 3x1 vector.
    """
    vg.shape.check(locals(), "translation", (3,))

    forward = np.eye(4)
    forward[:, -1][:-1] = translation

    if not ret_inverse_matrix:
        return forward

    inverse = np.eye(4)
    inverse[:, -1][:-1] = -translation
    return forward, inverse


def transform_matrix_for_non_uniform_scale(
    x_factor, y_factor, z_factor, allow_flipping=False, ret_inverse_matrix=False
):
    """
    Create a transformation matrix that scales by the given factors along
    `x`, `y`, and `z`.

    Forward:
        [[  s_0, 0,   0,   0 ],
        [  0,   s_1, 0,   0 ],
        [  0,   0,   s_2, 0 ],
        [  0,   0,   0,   1 ]]

    Reverse:
        [[  1/s_0, 0,     0,     0 ],
        [  0,     1/s_1, 0,     0 ],
        [  0,     0,     1/s_2, 0 ],
        [  0,     0,     0,     1 ]]

    Args:
        x_factor (float): The scale factor to be applied along the `x` axis,
            which should be positive.
        y_factor (float): The scale factor to be applied along the `y` axis,
            which should be positive.
        z_factor (float): The scale factor to be applied along the `z` axis,
            which should be positive.
        allow_flipping (bool): When `True`, allows scale factors to be
            positive or negative, though not zero.
        ret_inverse_matrix (bool): When `True`, also returns a matrix which
            provides the inverse transform.
    """
    if x_factor == 0 or y_factor == 0 or z_factor == 0:
        raise ValueError("Scale factors should be nonzero")
    if not allow_flipping and (x_factor < 0 or y_factor < 0 or z_factor < 0):
        raise ValueError("Scale factors should be greater than zero")
    scale = np.array([x_factor, y_factor, z_factor])

    forward = _convert_33_to_44(np.diag(scale))

    if not ret_inverse_matrix:
        return forward

    inverse = _convert_33_to_44(np.diag(1.0 / scale))
    return forward, inverse


def transform_matrix_for_uniform_scale(
    scale_factor, allow_flipping=False, ret_inverse_matrix=False
):
    """
    Create a transformation matrix that scales by the given factor.

    Forward:
        [[  s_0, 0,   0,   0 ],
        [  0,   s_1, 0,   0 ],
        [  0,   0,   s_2, 0 ],
        [  0,   0,   0,   1 ]]

    Reverse:
        [[  1/s_0, 0,     0,     0 ],
        [  0,     1/s_1, 0,     0 ],
        [  0,     0,     1/s_2, 0 ],
        [  0,     0,     0,     1 ]]

    Args:
        factor (float): The scale factor.
        ret_inverse_matrix (bool): When `True`, also returns a matrix which
            provides the inverse transform.
    """
    if scale_factor == 0:
        raise ValueError("Scale factor should be nonzero")
    if not allow_flipping and scale_factor < 0:
        raise ValueError("Scale factor should be greater than zero")
    return transform_matrix_for_non_uniform_scale(
        scale_factor,
        scale_factor,
        scale_factor,
        allow_flipping=allow_flipping,
        ret_inverse_matrix=ret_inverse_matrix,
    )
