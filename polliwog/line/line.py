from __future__ import print_function, absolute_import
import vg
from .functions import project_to_line


class Line:
    def __init__(self, point, along, assume_normalized=False):
        vg.shape.check(locals(), "point", (3,))
        vg.shape.check(locals(), "along", (3,))

        if vg.almost_zero(along):
            raise ValueError("along should not be the zero vector")

        self.reference_point = point
        self.along = along
        self.assume_normalized = assume_normalized

    @classmethod
    def from_points(cls, p1, p2):
        vg.shape.check(locals(), "p1", (3,))
        vg.shape.check(locals(), "p2", (3,))
        return cls(point=p1, along=p2 - p1)

    @property
    def reference_points(self):
        """
        Return two reference points on the line.
        """
        return self.reference_point, self.reference_point + self.along

    def intersect_line(self, other):
        """
        Find the intersection with another line.
        """
        from .line_intersect import line_intersect3

        return line_intersect3(*(self.reference_points + other.reference_points))

    def project(self, points):
        """
        Project a given point (or stack of points) to the plane.
        """
        return project_to_line(
            points=points,
            reference_points_of_lines=self.reference_point,
            vectors_along_lines=self.along,
        )
