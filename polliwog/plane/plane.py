import numpy as np
import vg


class Plane(object):
    """
    A 2-D plane in 3-space (not a hyperplane).

    Params:
        - point_on_plane, plane_normal:
            1 x 3 np.arrays
    """

    def __init__(self, point_on_plane, unit_normal):
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
        from .._temporary.coercion import as_numeric_array

        p1 = as_numeric_array(p1, shape=(3,))
        p2 = as_numeric_array(p2, shape=(3,))
        p3 = as_numeric_array(p3, shape=(3,))

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        return cls(point_on_plane=p1, unit_normal=normal)

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
        from .._temporary.coercion import as_numeric_array

        p1 = as_numeric_array(p1, shape=(3,))
        p2 = as_numeric_array(p2, shape=(3,))

        v1 = p2 - p1
        v2 = as_numeric_array(vector, shape=(3,))
        normal = np.cross(v1, v2)

        return cls(point_on_plane=p1, unit_normal=normal)

    @classmethod
    def fit_from_points(cls, points):
        """
        Fits a plane whose normal is orthgonal to the first two principal axes
        of variation in the data and centered on the points' centroid.
        """
        eigval, eigvec = np.linalg.eig(np.cov(points.T))
        ordering = np.argsort(eigval)[::-1]
        normal = np.cross(eigvec[:, ordering[0]], eigvec[:, ordering[1]])
        return cls(points.mean(axis=0), normal)

    @property
    def equation(self):
        """
        Returns parameters A, B, C, D as a 1 x 4 np.array, where

            Ax + By + Cz + D = 0

        defines the plane.

        params:
            - normalized:
                Boolean, indicates whether or not the norm of the vector [A, B, C] is 1.
                Useful when computing the distance from a point to the plane.
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
        normal = self._n * -1
        return Plane(self._r0, normal)

    def sign(self, points):
        """
        Given an array of points, return an array with +1 for points in front
        of the plane (in the direction of the normal), -1 for points behind
        the plane (away from the normal), and 0 for points on the plane.

        """
        return np.sign(self.signed_distance(points))

    def points_in_front(self, points, inverted=False, ret_indices=False):
        """
        Given an array of points, return the points which lie either on the
        plane or in the half-space in front of it (i.e. in the direction of
        the plane normal).

        points: An array of points.
        inverted: When `True`, invert the logic. Return the points that lie
          behind the plane instead.
        ret_indices: When `True`, return the indices instead of the points
          themselves.

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
        Returns the signed distances given an np.array of 3-vectors.

        Params:
            - points:
                V x 3 np.array
        """
        return np.dot(points, self.equation[:3]) + self.equation[3]

    def distance(self, points):
        return np.absolute(self.signed_distance(points))

    def project_point(self, point):
        """
        Project a given point to the plane.

        """
        # Translate the point back to the plane along the normal.
        signed_distance_to_point = self.signed_distance(point.reshape((-1, 3)))[0]
        return point - signed_distance_to_point * self._n

    def polyline_xsection(self, polyline, ret_edge_indices=False):
        """
        Deprecated.
        """
        return polyline.intersect_plane(self, ret_edge_indices=ret_edge_indices)

    def line_xsection(self, pt, ray):
        return self._line_xsection(np.asarray(pt).ravel(), np.asarray(ray).ravel())

    def _line_xsection(self, pt, ray):
        denom = np.dot(ray, self.normal)
        if denom == 0:
            return None  # parallel, either coplanar or non-intersecting
        p = np.dot(self.reference_point - pt, self.normal) / denom
        return p * ray + pt

    def line_segment_xsection(self, a, b):
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
