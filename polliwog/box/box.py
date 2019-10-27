import numpy as np
import vg


class Box(object):
    """
    An axis-aligned cuboid.

    """

    def __init__(self, origin, size):
        """
        origin: The x, y, z coordinate of the origin.
        size: A sequence containing the width (dx), height (dy), and
          depth (dz).

        """
        vg.shape.check(locals(), "origin", (3,))
        vg.shape.check(locals(), "size", (3,))
        if any(np.less(size, 0)):
            raise ValueError("Shape should be zero or positive")
        self.origin = origin
        self.size = size

    @classmethod
    def from_points(cls, points):
        k = vg.shape.check(locals(), "points", (-1, 3))
        if k == 0:
            raise ValueError("Need at least 1 point")
        return cls(np.min(points, axis=0), np.ptp(points, axis=0))

    @property
    def ranges(self):
        """
        Return ranges for each coordinate axis as a 3x2 numpy array.

        """
        ranges = np.array([self.origin, self.origin + self.size]).T
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
        return self.origin[0] + self.size[0]

    @property
    def max_y(self):
        return self.origin[1] + self.size[1]

    @property
    def max_z(self):
        return self.origin[2] + self.size[2]

    @property
    def mid_x(self):
        return self.origin[0] + self.size[0] / 2

    @property
    def mid_y(self):
        return self.origin[1] + self.size[1] / 2

    @property
    def mid_z(self):
        return self.origin[2] + self.size[2] / 2

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def depth(self):
        return self.size[2]

    @property
    def center_point(self):
        return self.origin + 0.5 * self.size

    @property
    def floor_point(self):
        return self.origin + [0.5, 0.0, 0.5] * self.size

    @property
    def volume(self):
        return np.prod(self.size)

    @property
    def surface_area(self):
        l, h, w = self.size
        return 2 * (w * l + h * l + h * w)
