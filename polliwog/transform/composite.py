import numpy as np
from blmath.numerics import vx

def convert_33_to_44(matrix):
    '''
    Transform from:
        array([[1., 2., 3.],
               [2., 3., 4.],
               [5., 6., 7.]])
    to:
        array([[1., 2., 3., 0.],
               [2., 3., 4., 0.],
               [5., 6., 7., 0.],
               [0., 0., 0., 1.]])

    '''
    if matrix.shape != (3, 3):
        raise ValueError("Expected 3x3 matrix, got: %s" % matrix.shape)
    result = np.pad(matrix, ((0, 1), (0, 1)), mode='constant')
    result[3][3] = 1
    return result

def convert_44_to_33(matrix):
    '''
    Transform from:
        array([[1., 2., 3., 0.],
               [2., 3., 4., 0.],
               [5., 6., 7., 0.],
               [0., 0., 0., 1.]])
    to:
        array([[1., 2., 3.],
               [2., 3., 4.],
               [5., 6., 7.]])

    '''
    if matrix.shape != (4, 4):
        raise ValueError("Expected 4x4 matrix, got: %s" % matrix.shape)
    return matrix[:3, :3]


class CompositeTransform(object):
    '''
    Composite transform using homogeneous coordinates.


    Example usage
    -------------

        transform = CompositeTransform()
        transform.scale(10)
        transform.reorient(up=[0, 1, 0], look=[-1, 0, 0])
        transform.translate([0, -2.5, 0])
        transformed_scan = Mesh(v=transform(scan.v), f=scan.f)

        ...

        untransformed_alignment = Mesh(
            v=transform(alignment.v, reverse=True),
            f=alignment.f
        )

    Backround
    ---------

    - Computer Graphics: Principles and Practice, Hughes, van Dam, McGuire,
      Sklar, Foley

    - http://gamedev.stackexchange.com/questions/72044/why-do-we-use-4x4-matrices-to-transform-things-in-3d

    '''

    def __init__(self):
        # List of tuples, containing forward and reverse matrices.
        self.transforms = []

    def __call__(self, points, from_range=None, reverse=False):
        '''
        points: Points to transform, as a 3xn array.
        range: The indices of the subset of the transformations to apply.
          e.g. (0, 2), (2, 4). The default is to apply them all.
        reverse: When `True` applies the selected transformations in reverse.
          This has no effect on how range is interpreted, only whether the
          selected transformations apply in the forward or reverse mode.

        '''
        matrix = self.matrix_for(from_range=from_range, reverse=reverse)

        return vx.unpad(np.dot(matrix, vx.pad_with_ones(points).T).T)

    def matrix_for(self, from_range=None, reverse=False):
        '''
        Return a 4x4 transformation matrix representation.

        range: The min and max indices of the subset of the transformations to
          apply. e.g. (0, 2), (2, 4). Inclusive of the min value, exclusive of
          the max value. The default is to apply them all.
        reverse: When `True` returns a matrix for the inverse transform.
          This has no effect on how range is interpreted, only whether the
          forward or reverse matrices are used.

        '''
        if from_range is not None:
            start, stop = from_range  # from_range is defined as None, a non-sequence, but when it's not None, it's always a sequence. pylint: disable=unpacking-non-sequence
            selected_transforms = self.transforms[start:stop]
        else:
            selected_transforms = self.transforms

        # The transpose of a product of matrices equals the products of each
        # transpose in reverse order.
        matrices = [
            reverse_matrix if reverse else forward_matrix.T
            for forward_matrix, reverse_matrix in selected_transforms
        ]

        if not len(matrices): # pylint: disable=len-as-condition
            return np.eye(4)

        matrix = reduce(np.dot, matrices)
        return matrix if reverse else matrix.T

    def append_transform4(self, forward, reverse=None):
        '''
        Append an arbitrary transformation, defined by 4x4 forward and reverse
        matrices.

        The new transformation is added to the end. Return its index.

        '''
        if reverse is None:
            reverse = np.linalg.inv(forward)

        new_index = len(self.transforms)

        self.transforms.append((forward, reverse))

        return new_index

    def append_transform3(self, forward, reverse=None):
        '''
        Append an arbitrary transformation, defined by 3x3 forward and reverse
        matrices.

        The new transformation is added to the end. Return its index.

        '''
        args = (forward,) if reverse is None else (forward, reverse)
        return self.append_transform4(*map(convert_33_to_44, args))

    def scale(self, factor):
        '''
        Scale by the given factor.

        factor: A float or int.

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

        '''
        if factor <= 0:
            raise ValueError('Scale factor should be greater than zero')

        forward3 = np.diag(np.repeat(factor, 3))
        reverse3 = np.diag(np.repeat(1./factor, 3))

        return self.append_transform3(forward3, reverse3)

    def convert_units(self, from_units, to_units):
        '''
        Convert the mesh from one set of units to another.

        These calls are equivalent:

        - composite.convert_units(from_units='cm', to_units='m')
        - composite.scale(.01)

        '''
        from blmath import units
        factor = units.factor(
            from_units=from_units,
            to_units=to_units,
            units_class='length'
        )
        self.scale(factor)

    def translate(self, vector):
        '''
        Translate by the vector provided.

        vector: A 3x1 vector.

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

        '''
        vector = np.asarray(vector)

        forward = np.eye(4)
        forward[:, -1][:-1] = vector

        reverse = np.eye(4)
        reverse[:, -1][:-1] = -vector

        return self.append_transform4(forward, reverse)

    def reorient(self, up, look):
        '''
        Reorient using up and look.

        '''
        from blmath.geometry.transform import rotation_from_up_and_look
        forward3 = rotation_from_up_and_look(up, look)
        # The inverse of a rotation matrix is its transpose.
        return self.append_transform3(forward3, forward3.T)

    def rotate(self, rot):
        '''
        Rotate by either an explicit matrix or a rodrigues vector
        '''
        from blmath.geometry.transform.rodrigues import as_rotation_matrix
        rot = np.asarray(rot)
        rot = as_rotation_matrix(rot)
        # The inverse of a rotation matrix is its transpose.
        return self.append_transform3(rot, rot.T)
