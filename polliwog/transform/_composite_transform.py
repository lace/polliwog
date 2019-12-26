import numpy as np
import vg
from ._affine_transform import (
    apply_affine_transform,
    transform_matrix_for_rotation,
    transform_matrix_for_scale,
    transform_matrix_for_translation,
)
from ._rotation import rotation_from_up_and_look


class CompositeTransform(object):
    """
    Composite transform using homogeneous coordinates.

    Example:
        >>> transform = CompositeTransform()
        >>> transform.scale(10)
        >>> transform.reorient(up=[0, 1, 0], look=[-1, 0, 0])
        >>> transform.translate([0, -2.5, 0])
        >>> transformed_scan = transform(scan_v)
        >>> # ... register the scan here ...
        >>> untransformed_alignment = transform(alignment_v, reverse=True)

    See also:

        - *Computer Graphics: Principles and Practice*, Hughes, van Dam,
          McGuire, Sklar, Foley
        - http://gamedev.stackexchange.com/questions/72044/why-do-we-use-4x4-matrices-to-transform-things-in-3d
    """

    def __init__(self):
        # List of tuples, containing forward and reverse matrices.
        self.transforms = []

    def __call__(self, points, from_range=None, reverse=False):
        """
        Args:
            points (np.arraylike): Points to transform, as a 3xn array.
            from_range (tuple): The indices of the subset of the
                transformations to apply. e.g. `(0, 2)`, `(2, 4)`. When
                `None`, which is the default, apply them all.
            reverse (bool): When `True` applies the selected transformations
                in reverse. This has no effect on how range is interpreted,
                only whether the selected transformations apply in the forward
                or reverse mode.

        """
        transform_matrix = self.transform_matrix_for(
            from_range=from_range, reverse=reverse
        )
        return apply_affine_transform(points=points, transform_matrix=transform_matrix)

    def transform_matrix_for(self, from_range=None, reverse=False):
        """
        Return a 4x4 transformation matrix representation.

        range: The min and max indices of the subset of the transformations to
          apply. e.g. (0, 2), (2, 4). Inclusive of the min value, exclusive of
          the max value. The default is to apply them all.
        reverse: When `True` returns a matrix for the inverse transform.
          This has no effect on how range is interpreted, only whether the
          forward or reverse matrices are used.

        """
        from functools import reduce

        if from_range is not None:
            start, stop = from_range
            selected_transforms = self.transforms[start:stop]
        else:
            selected_transforms = self.transforms

        # The transpose of a product of matrices equals the products of each
        # transpose in reverse order.
        matrices = [
            reverse_matrix if reverse else forward_matrix.T
            for forward_matrix, reverse_matrix in selected_transforms
        ]

        if not len(matrices):
            return np.eye(4)

        matrix = reduce(np.dot, matrices)
        return matrix if reverse else matrix.T

    def append_transform(self, forward, reverse=None):
        """
        Append an arbitrary transformation, defined by 4x4 forward and reverse
        matrices.

        The new transformation is added to the end. Return its index.

        """
        vg.shape.check(locals(), "forward", (4, 4))
        if reverse is None:
            reverse = np.linalg.inv(forward)
        else:
            vg.shape.check(locals(), "reverse", (4, 4))

        new_index = len(self.transforms)
        self.transforms.append((forward, reverse))
        return new_index

    def scale(self, factor):
        """
        Scale by the given factor.

        Args:
            factor (float): The scale factor.
        """
        forward, inverse = transform_matrix_for_scale(factor, ret_inverse_matrix=True)
        return self.append_transform(forward, inverse)

    def convert_units(self, from_units, to_units):
        """
        Convert the mesh from one set of units to another.

        These calls are equivalent:

        >>> composite.convert_units(from_units='cm', to_units='m')
        >>> composite.scale(.01)

        Supports the length units from Ounce:
        https://github.com/lace/ounce/blob/master/ounce/core.py#L26
        """
        import ounce

        factor = ounce.factor(from_units, to_units)
        return self.scale(factor)

    def translate(self, translation):
        """
        Translate by the vector provided.

        Args:
            vector (np.arraylike): A 3x1 vector.
        """
        forward, inverse = transform_matrix_for_translation(
            translation, ret_inverse_matrix=True
        )
        return self.append_transform(forward, inverse)

    def reorient(self, up, look):
        """
        Reorient using up and look.
        """
        return self.rotate(rotation_from_up_and_look(up, look))

    def rotate(self, rotation):
        """
        Rotate by the given 3x3 rotation matrix or a Rodrigues vector.
        """
        forward, inverse = transform_matrix_for_rotation(
            rotation, ret_inverse_matrix=True
        )
        return self.append_transform(forward, inverse)
