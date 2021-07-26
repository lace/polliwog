import numpy as np
from vg.compat import v2 as vg
from ._plane_functions import (
    mirror_point_across_plane,
    plane_normal_from_points,
    project_point_to_plane,
    signed_distance_to_plane,
)


class Plane(object):
    """
    A 2-D plane in 3-space (not a hyperplane).

    Args:
        point_on_plane (np.arraylike): A reference point on the plane, as a
            NumPy array with three coordinates.
        unit_normal (np.arraylike): The plane normal vector, as a NumPy
            array with three coordinates.
    """

    def __init__(self, point_on_plane, unit_normal):
        vg.shape.check(locals(), "point_on_plane", (3,))
        vg.shape.check(locals(), "unit_normal", (3,))

        if vg.almost_zero(unit_normal):
            raise ValueError("unit_normal should not be the zero vector")

        unit_normal = vg.normalize(unit_normal)

        self._r0 = np.asarray(point_on_plane)
        self._n = np.asarray(unit_normal)

    def __repr__(self):
        return "<Plane of {} through {}>".format(self.normal, self.reference_point)

    @classmethod
    def from_points(cls, p1, p2, p3):
        """
        If the points are oriented in a counterclockwise direction, the plane's
        normal extends towards you.

        """
        vg.shape.check(locals(), "p1", (3,))
        vg.shape.check(locals(), "p2", (3,))
        vg.shape.check(locals(), "p3", (3,))
        points = np.array([p1, p2, p3])
        return cls(point_on_plane=p1, unit_normal=plane_normal_from_points(points))

    @classmethod
    def from_points_and_vector(cls, p1, p2, vector):
        """
        Compute a plane which contains two given points and the given
        vector. Its reference point will be p1.

        For example, to find the vertical plane that passes through
        two landmarks:

            from_points_and_normal(p1, p2, vector)

        Another way to think about this: identify the plane to which
        your result plane should be perpendicular, and specify vector
        as its normal vector.

        """
        vg.shape.check(locals(), "p1", (3,))
        vg.shape.check(locals(), "p2", (3,))
        vg.shape.check(locals(), "vector", (3,))

        normal = np.cross(p2 - p1, vector)

        return cls(point_on_plane=p1, unit_normal=normal)

    @classmethod
    def fit_from_points(cls, points):
        """
        Fits a plane whose normal is orthgonal to the first two principal axes
        of variation in the data and centered on their centroid.
        """
        vg.shape.check(locals(), "points", (-1, 3))

        eigval, eigvec = np.linalg.eig(np.cov(points.T))
        ordering = np.argsort(eigval)[::-1]
        normal = np.cross(eigvec[:, ordering[0]], eigvec[:, ordering[1]])

        centroid = points.mean(axis=0)

        return cls(centroid, normal)

    @property
    def equation(self):
        """
        Returns parameters `A`, `B`, `C`, `D` as a 1x4 `np.array`, where

            `Ax + By + Cz + D = 0`

        defines the plane.
        """
        A, B, C = self._n
        D = -self._r0.dot(self._n)

        return np.array([A, B, C, D])

    @property
    def reference_point(self):
        """
        The point used to create this plane.

        """
        return self._r0

    @property
    def canonical_point(self):
        """
        A canonical point on the plane, the one at which the normal
        would intersect the plane if drawn from the origin (0, 0, 0).

        This is computed by projecting the reference point onto the
        normal.

        This is useful for partitioning the space between two planes,
        as we do when searching for planar cross sections.

        """
        return self._r0.dot(self._n) * self._n

    @property
    def normal(self):
        """
        Return the plane's normal vector.

        """
        return self._n

    def flipped(self):
        """
        Creates a new Plane with an inverted orientation.
        """
        return Plane(point_on_plane=self._r0, unit_normal=-self._n)

    def sign(self, points):
        """
        Given an array of points, return an array with +1 for points in front
        of the plane (in the direction of the normal), -1 for points behind
        the plane (away from the normal), and 0 for points on the plane.

        """
        return np.sign(self.signed_distance(points))

    def points_in_front(self, points, inverted=False, ret_indices=False):
        """
        Given an array of points, return the points which lie in the
        half-space in front of it (i.e. in the direction of the plane
        normal).

        Args:
            points (np.arraylikw): An array of points.
            inverted (bool): When `True`, return the points which lie on or
                behind the plane instead.
            ret_indices (bool): When `True`, return the indices instead of the
                points themselves.

        Note:
            Use `points_on_or_in_front()` for points which lie either on the
            plane or in front of it.
        """
        sign = self.sign(points)

        if inverted:
            mask = np.less(sign, 0)
        else:
            mask = np.greater(sign, 0)

        indices = np.flatnonzero(mask)

        return indices if ret_indices else points[indices]

    def points_on_or_in_front(self, points, inverted=False, ret_indices=False):
        """
        Given an array of points, return the points which lie either on the
        plane or in the half-space in front of it (i.e. in the direction of
        the plane normal).

        Args:
            points (np.arraylikw): An array of points.
            inverted (bool): When `True`, return the points behind the plane
                instead.
            ret_indices (bool): When `True`, return the indices instead of the
                points themselves.

        Note:
            Use `points_in_front()` to get points which lie only in front of
            the plane.
        """
        sign = self.sign(points)

        if inverted:
            mask = np.less_equal(sign, 0)
        else:
            mask = np.greater_equal(sign, 0)

        indices = np.flatnonzero(mask)

        return indices if ret_indices else points[indices]

    def signed_distance(self, points):
        """
        Returns the signed distances to the given points or the signed
        distance to a single point.

        Args:
            points (np.arraylike): A 3D point or a `kx3` stack of points.

        Returns:
            depends:

            - Given a single 3D point, the distance as a NumPy scalar.
            - Given a `kx3` stack of points, an `k` array of distances.
        """
        return signed_distance_to_plane(points, self.equation)

    def distance(self, points):
        return np.absolute(self.signed_distance(points))

    def project_point(self, points):
        """
        Project a given point (or stack of points) to the plane.
        """
        return project_point_to_plane(points, self.equation)

    def mirror_point(self, points):
        """
        Mirror a point (or stack of points) to the opposite side of the plane.
        """
        return mirror_point_across_plane(points, self.equation)

    def line_xsection(self, pt, ray):
        vg.shape.check(locals(), "pt", (3,))
        vg.shape.check(locals(), "ray", (3,))
        return self._line_xsection(np.asarray(pt).ravel(), np.asarray(ray).ravel())

    def _line_xsection(self, pt, ray):
        denom = np.dot(ray, self.normal)
        if denom == 0:
            return None  # parallel, either coplanar or non-intersecting
        p = np.dot(self.reference_point - pt, self.normal) / denom
        return p * ray + pt

    def line_segment_xsection(self, a, b):
        vg.shape.check(locals(), "a", (3,))
        vg.shape.check(locals(), "b", (3,))
        return self._line_segment_xsection(np.asarray(a).ravel(), np.asarray(b).ravel())

    def _line_segment_xsection(self, a, b):
        pt = self._line_xsection(a, b - a)
        if pt is not None:
            if any(np.logical_and(pt > a, pt > b)) or any(
                np.logical_and(pt < a, pt < b)
            ):
                return None
        return pt

    def line_xsections(self, pts, rays):
        k = vg.shape.check(locals(), "pts", (-1, 3))
        vg.shape.check(locals(), "rays", (k, 3))
        denoms = np.dot(rays, self.normal)
        denom_is_zero = denoms == 0
        denoms[denom_is_zero] = np.nan
        p = np.dot(self.reference_point - pts, self.normal) / denoms
        return np.vstack([p, p, p]).T * rays + pts, ~denom_is_zero

    def line_segment_xsections(self, a, b):
        pts, pt_is_valid = self.line_xsections(a, b - a)
        pt_is_out_of_bounds = np.logical_or(
            np.any(
                np.logical_and(
                    pts[pt_is_valid] > a[pt_is_valid], pts[pt_is_valid] > b[pt_is_valid]
                ),
                axis=1,
            ),
            np.any(
                np.logical_and(
                    pts[pt_is_valid] < a[pt_is_valid], pts[pt_is_valid] < b[pt_is_valid]
                ),
                axis=1,
            ),
        )
        pt_is_valid[pt_is_valid] = ~pt_is_out_of_bounds
        pts[~pt_is_valid] = np.nan
        return pts, pt_is_valid

    def tilted(self, new_point, coplanar_point):
        """
        Create a new plane, tilted so it passes through `new_point`. Also
        specify a `coplanar_point` which the old and new planes should have
        in common.

        Args:
            new_point (np.arraylike): A point on the desired plane, with shape
                `(3,)`.
            coplanar_point (np.arraylike): The `(3,)` point which the old and
                new planes have in common.

        Returns:
            Plane: The adjusted plane.
        """
        vg.shape.check(locals(), "new_point", (3,))
        vg.shape.check(locals(), "coplanar_point", (3,))

        vector_along_old_plane = self.project_point(new_point) - coplanar_point
        vector_along_new_plane = new_point - coplanar_point
        axis_of_rotation = vg.perpendicular(vector_along_old_plane, self.normal)
        angle_between_vectors = vg.signed_angle(
            vector_along_old_plane,
            vector_along_new_plane,
            look=axis_of_rotation,
            units="rad",
        )
        new_normal = vg.rotate(
            self.normal,
            around_axis=axis_of_rotation,
            angle=angle_between_vectors,
            units="rad",
        )
        return Plane(point_on_plane=coplanar_point, unit_normal=new_normal)


Plane.xy = Plane(point_on_plane=np.zeros(3), unit_normal=vg.basis.z)
Plane.xy.__doc__ = "The `xy`-plane."
Plane.xz = Plane(point_on_plane=np.zeros(3), unit_normal=vg.basis.y)
Plane.xz.__doc__ = "The `xz`-plane."
Plane.yz = Plane(point_on_plane=np.zeros(3), unit_normal=vg.basis.x)
Plane.yz.__doc__ = "The `yz`-plane."
