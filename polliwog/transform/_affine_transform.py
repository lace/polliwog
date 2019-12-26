import numpy as np
import vg
from .._common.shape import columnize

__all__ = [
    "apply_affine_transform",
    "transform_matrix_for_rotation",
    "transform_matrix_for_translation",
    "transform_matrix_for_scale",
]


def apply_affine_transform(points, transform_matrix):
    """
    Apply the given transformation matrix to the points using homogenous
    coordinates.

    Note:
        This works on any transformation matrix, whether or not it is affine.
    """
    vg.shape.check(locals(), "transform_matrix", (4, 4))
    points, _, maybe_decolumnize = columnize(points, (-1, 3), name="points")

    padded_points = np.pad(points, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    transformed_padded_points = np.dot(transform_matrix, padded_points.T).T
    transformed_points = np.delete(transformed_padded_points, 3, axis=1)

    return maybe_decolumnize(transformed_points)


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


def transform_matrix_for_scale(scale_factor, ret_inverse_matrix=False):
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
    if scale_factor <= 0:
        raise ValueError("Scale factor should be greater than zero")

    forward = _convert_33_to_44(np.diag(np.repeat(scale_factor, 3)))

    if not ret_inverse_matrix:
        return forward

    inverse = _convert_33_to_44(np.diag(np.repeat(1.0 / scale_factor, 3)))
    return forward, inverse
