import numpy as np
import vg
from .._common.shape import columnize


def apply_affine_transform(points, transform_matrix):
    """
    Apply the given transformation matrix to the points using homogenous
    coordinates.

    (This works on any transformation matrix, whether or not it is affine.)
    """
    vg.shape.check(locals(), "transform_matrix", (4, 4))
    points, _, maybe_decolumnize = columnize(points, (-1, 3), name="points")

    padded_points = np.pad(points, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    transformed_padded_points = np.dot(transform_matrix, padded_points.T).T
    transformed_points = np.delete(transformed_padded_points, 3, axis=1)

    return maybe_decolumnize(transformed_points)
