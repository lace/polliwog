import numpy as np
import vg
from .._common.shape import columnize

__all__ = [
    "apply_transform",
]


def apply_transform(transform):
    """
    Wrap the given transformation matrix with a function which conveniently can
    be invoked with either points or a single point, returning the same. It
    applies the transformation to those points using homogeneous coordinates.

    Args:
        points (np.ndarray): The point `(3,)` or points `kx3` to transform.

    Return:
        func: A function which accepts an `np.ndarray` containing a point
        `(3,)` or points `kx3` to transform, and returns an `ndarray` of the
        same shape. Also accepts a kwarg `discard_z_coord`. when `True`,
        discard the z coordinate of the result. This is useful when applying
        viewport transformations.
    """
    vg.shape.check(locals(), "transform", (4, 4))

    def apply(points, discard_z_coord=False):
        points, is_columnized, maybe_decolumnize = columnize(
            points, (-1, 3), name="points"
        )

        padded_points = np.pad(
            points, ((0, 0), (0, 1)), mode="constant", constant_values=1
        )
        transformed_padded_points = np.dot(transform, padded_points.T).T
        transformed_points = np.delete(transformed_padded_points, 3, axis=1)

        result = maybe_decolumnize(transformed_points)
        if discard_z_coord:
            return result[:, 0:2] if is_columnized else result[0:2]
        else:
            return result

    return apply


def compose_transforms(*transforms):
    """
    Compose the provided transformation matrices in order, returning a composite
    transformation.

    Args:
        transforms (list): One or more `4x4` transformation matrices.

    Return:
        np.ndarray: A `4x4` transformation matrix.
    """
    from functools import reduce

    for transform in transforms:
        vg.shape.check(locals(), "transform", (4, 4))

    if len(transforms) == 0:
        return np.eye(4)

    return reduce(np.dot, reversed(transforms))
