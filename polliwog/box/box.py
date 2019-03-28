import numpy as np

class Box(object):
    '''
    An axis-aligned cuboid.

    '''
    def __init__(self, origin, shape):
        '''
        origin: The x, y, z coordinate of the origin.
        shape: A sequence containing the width (dx), height (dy), and
          depth (dz).

        '''
        from blmath.numerics import as_numeric_array
        self.origin = as_numeric_array(origin, shape=(3,))
        self.shape = as_numeric_array(shape, shape=(3,))
        if any(np.less(self.shape, 0)):
            raise ValueError('Shape should be zero or positive')

    @property
    def ranges(self):
        '''
        Return ranges for each coordinate axis as a 3x2 numpy array.

        '''
        ranges = np.array([self.origin, self.origin + self.shape]).T
        # ranges is almost, but not quite what we want, since it might
        # include mins which are greater than maxes, and vice versa.
        return np.vstack([ranges.min(axis=1), ranges.max(axis=1)]).T

    @property
    def min_x(self):
        return self.origin[0]

    @property
    def min_y(self):
        return self.origin[1]

    @property
    def min_z(self):
        return self.origin[2]

    @property
    def max_x(self):
        return self.origin[0] + self.shape[0]

    @property
    def max_y(self):
        return self.origin[1] + self.shape[1]

    @property
    def max_z(self):
        return self.origin[2] + self.shape[2]

    @property
    def mid_x(self):
        return self.origin[0] + self.shape[0] / 2

    @property
    def mid_y(self):
        return self.origin[1] + self.shape[1] / 2

    @property
    def mid_z(self):
        return self.origin[2] + self.shape[2] / 2

    @property
    def width(self):
        return self.shape[0]

    @property
    def height(self):
        return self.shape[1]

    @property
    def depth(self):
        return self.shape[2]

    @property
    def center_point(self):
        return self.origin + 0.5 * self.shape

    @property
    def floor_point(self):
        return self.origin + [0.5, 0., 0.5] * self.shape

    @property
    def volume(self):
        return np.prod(self.shape)

    @property
    def surface_area(self):
        l, h, w = self.shape # self.shape is a np.ndarray, which is a sequence. pylint: disable=unpacking-non-sequence
        return 2 * (w * l + h * l + h * w)
