import numpy as np
from vg.compat import v2 as vg
from ..plane._plane_object import Plane


class Box(object):
    """
    An axis-aligned cuboid or rectangular prism. It's defined by an origin
    point, which is its minimum point in each dimension, and non-negative size
    (length, width, and depth).

    Args:
        origin (np.arraylike): The `x`, `y`, and `z` coordinate of the
            origin, the minimum point in each dimension.
        size (np.arraylike): An array containing the width (dx), height
            (dy), and depth (dz), which must be non-negative.
    """

    def __init__(self, origin, size):
        vg.shape.check(locals(), "origin", (3,))
        vg.shape.check(locals(), "size", (3,))
        if any(np.less(size, 0)):
            raise ValueError("Shape should be zero or positive")
        self.origin = origin
        self.size = size

    @classmethod
    def from_points(cls, points):
        """
        The smallest box which spans the given points.

        Args:
            points (np.arraylike): A `kx3` array of points.

        Returns:
            Box: The smallest box which spans the given points.
        """
        k = vg.shape.check(locals(), "points", (-1, 3))
        if k == 0:
            raise ValueError("Need at least 1 point")
        return cls(np.min(points, axis=0), np.ptp(points, axis=0))

    @property
    def ranges(self):
        """
        Ranges for each coordinate axis as a 3x2 `np.ndarray`.
        """
        ranges = np.array([self.origin, self.origin + self.size]).T
        # ranges is almost, but not quite what we want, since it might
        # include mins which are greater than maxes, and vice versa.
        # TODO: Is this really true? `self.size` is nonnegative...
        return np.vstack([ranges.min(axis=1), ranges.max(axis=1)]).T

    @property
    def min_x(self):
        """
        The box's minimum `x` coordinate.
        """
        return self.origin[0]

    @property
    def min_y(self):
        """
        The box's minimum `y` coordinate.
        """
        return self.origin[1]

    @property
    def min_z(self):
        """
        The box's minimum `z` coordinate.
        """
        return self.origin[2]

    @property
    def max_x(self):
        """
        The box's maximum `x` coordinate.
        """
        return self.origin[0] + self.size[0]

    @property
    def max_y(self):
        """
        The box's maximum `y` coordinate.
        """
        return self.origin[1] + self.size[1]

    @property
    def max_z(self):
        """
        The box's maximum `z` coordinate.
        """
        return self.origin[2] + self.size[2]

    @property
    def mid_x(self):
        """
        The `x` coordinate of the box's center.
        """
        return self.origin[0] + self.size[0] / 2

    @property
    def mid_y(self):
        """
        The `y` coordinate of the box's center.
        """
        return self.origin[1] + self.size[1] / 2

    @property
    def mid_z(self):
        """
        The `z` coordinate of the box's center.
        """
        return self.origin[2] + self.size[2] / 2

    @property
    def min_x_plane(self):
        """
        The plane facing the inside of the box, aligned with its minimum
        `x` coordinate.
        """
        center_of_side = self.center_point
        center_of_side[0] = self.min_x
        return Plane(center_of_side, vg.basis.x)

    @property
    def min_y_plane(self):
        """
        The plane facing the inside of the box, aligned with its minimum
        `y` coordinate.
        """
        center_of_side = self.center_point
        center_of_side[1] = self.min_y
        return Plane(center_of_side, vg.basis.y)

    @property
    def min_z_plane(self):
        """
        The plane facing the inside of the box, aligned with its minimum
        `z` coordinate.
        """
        center_of_side = self.center_point
        center_of_side[2] = self.min_z
        return Plane(center_of_side, vg.basis.z)

    @property
    def max_x_plane(self):
        """
        The plane facing the inside of the box, aligned with its maximum
        `x` coordinate.
        """
        center_of_side = self.center_point
        center_of_side[0] = self.max_x
        return Plane(center_of_side, vg.basis.neg_x)

    @property
    def max_y_plane(self):
        """
        The plane facing the inside of the box, aligned with its maximum
        `y` coordinate.
        """
        center_of_side = self.center_point
        center_of_side[1] = self.max_y
        return Plane(center_of_side, vg.basis.neg_y)

    @property
    def max_z_plane(self):
        """
        The plane facing the inside of the box, aligned with its maximum
        `z` coordinate.
        """
        center_of_side = self.center_point
        center_of_side[2] = self.max_z
        return Plane(center_of_side, vg.basis.neg_z)

    @property
    def width(self):
        """
        The box's width. Same as `max_x - min_x`.
        """
        return self.size[0]

    @property
    def height(self):
        """
        The box's height. Same as `max_y - min_y`.
        """
        return self.size[1]

    @property
    def depth(self):
        """
        The box's depth. Same as `max_z - min_z`.
        """
        return self.size[2]

    @property
    def center_point(self):
        """
        The box's geometric center.
        """
        return self.origin + 0.5 * self.size

    @property
    def floor_point(self):
        """
        The center of the side of the box having the minimum `y` coordinate.
        This is `center_point` projected to the the level of `min_y`.
        """
        return self.origin + [0.5, 0.0, 0.5] * self.size

    @property
    def volume(self):
        """
        The box's volume.
        """
        return np.prod(self.size)

    @property
    def surface_area(self):
        """
        The box's surface area.
        """
        l, h, w = self.size
        return 2 * (w * l + h * l + h * w)

    @property
    def v(self):
        """
        Corners of the box as an `8x3` array of coordinates.
        """
        return np.array(
            [
                self.origin,
                self.origin + np.array([self.size[0], 0, 0]),
                self.origin + np.array([0, self.size[1], 0]),
                self.origin + np.array([0, 0, self.size[2]]),
                self.origin + np.array([self.size[0], self.size[1], 0]),
                self.origin + np.array([0, self.size[1], self.size[2]]),
                self.origin + np.array([self.size[0], 0, self.size[2]]),
                self.origin + np.array([self.size[0], self.size[1], self.size[2]]),
            ]
        )

    def contains(self, point, atol=None):
        """
        Test whether the box contains the given point. When `atol` is
        provided, returns `True` for points inside the box and points
        whose coordinates are all within `atol` of the box boundary.
        """
        vg.shape.check(locals(), "point", (3,))

        if atol is None:
            atol = 0.0
        return np.all(
            np.logical_and(
                self.origin - atol <= point, point <= self.origin + self.size + atol
            )
        )
