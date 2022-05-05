import numpy as np
from vg.compat import v2 as vg
from ._plane_functions import (
    mirror_point_across_plane,
    plane_normal_from_points,
    project_point_to_plane,
    signed_distance_to_plane,
)


class Plane:
    """
    An immutable 2D plane (not a hyperplane) in 3D space.

    Args:
        reference_point (np.ndarray): A reference point on the plane, as a
            NumPy array with three coordinates.
        normal (np.ndarray): The plane normal vector, with unit length, as a
            NumPy array with three coordinates.
        direction_decimals (int): The desired number of decimal places for
            validating that `normal` has unit length. The default is
            `DEFAULT_DIRECTION_DECIMALS`.

    Note:
        To construct a plane from a non-unit length normal, use
        `Plane.from_point_and_normal()`.
    """

    POSITION_DTYPE = np.float64
    DEFAULT_POSITION_DECIMALS = 6
    DEFAULT_DIRECTION_DECIMALS = 6

    def __init__(self, reference_point, normal, direction_decimals=None):
        if direction_decimals is None:
            direction_decimals = self.DEFAULT_DIRECTION_DECIMALS

        vg.shape.check(locals(), "reference_point", (3,))
        vg.shape.check(locals(), "normal", (3,))

        if not vg.almost_unit_length(normal, atol=0.1**direction_decimals):
            raise ValueError("normal should have unit length")

        self.reference_point = np.copy(reference_point)
        self.reference_point.setflags(write=False)

        self.normal = np.copy(normal)
        self.normal.setflags(write=False)

    def __repr__(self):
        return "<Plane of {} through {}>".format(self.normal, self.reference_point)

    @classmethod
    def from_point_and_normal(cls, reference_point, normal, direction_decimals=None):
        """
        Create a plane using the given reference point and normal vector, which
        will be normalized for you.

        Args:
            reference_point (np.ndarray): A reference point on the plane, as a
                NumPy array with three coordinates.
            normal (np.ndarray): The plane normal vector, with unit length, as a
                NumPy array with three coordinates.
            direction_decimals (int): The desired number of decimal places for
                validating that `normal` has unit length. The default is
                `DEFAULT_DIRECTION_DECIMALS`.

        Returns:
            Plane: The requested plane.
        """
        return cls(
            reference_point=reference_point,
            normal=vg.normalize(normal),
            direction_decimals=direction_decimals,
        )

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
        return cls(reference_point=p1, normal=plane_normal_from_points(points))

    @classmethod
    def from_points_and_vector(cls, p1, p2, vector, direction_decimals=None):
        """
        Compute a plane which contains two given points and the given
        vector. Its reference point will be p1.

        For example, to find the vertical plane that passes through
        two landmarks:

            from_points_and_normal(p1, p2, vector)

        Another way to think about this: identify the plane to which
        your result plane should be perpendicular, and specify vector
        as its normal vector.

        Args:
            direction_decimals (int): The desired number of decimal places for
                validating that `normal` has unit length. The default is
                `DEFAULT_DIRECTION_DECIMALS`.

        """
        vg.shape.check(locals(), "p1", (3,))
        vg.shape.check(locals(), "p2", (3,))
        vg.shape.check(locals(), "vector", (3,))

        return cls.from_point_and_normal(
            reference_point=p1,
            normal=np.cross(p2 - p1, vector),
            direction_decimals=direction_decimals,
        )

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

    def rounded(self, position_decimals=None, direction_decimals=None):
        """
        Return a copy of this plane, with the reference point and normal rounded
        to the specified precision.

        Args:
            position_decimals (int): The desired number of decimal places for
                the reference point. The default is `DEFAULT_POSITION_DECIMALS`.
            direction_decimals (int): The desired number of decimal places for
                the normal. The default is `DEFAULT_DIRECTION_DECIMALS`.

        Returns:
            Plane: The rounded plane.
        """
        if position_decimals is None:
            position_decimals = self.DEFAULT_POSITION_DECIMALS
        if direction_decimals is None:
            direction_decimals = self.DEFAULT_DIRECTION_DECIMALS
        return Plane(
            reference_point=np.around(self.reference_point, position_decimals),
            normal=np.around(self.normal, direction_decimals),
        )

    def serialize(self, position_decimals=None, direction_decimals=None):
        """
        Return a JSON representation of this plane, with the reference point and
        normal rounded to the specified precision.

        The schema is defined in `types/src/schema.json`.

        Args:
            position_decimals (int): The desired number of decimal places for
                the reference point. The default is `DEFAULT_POSITION_DECIMALS`.
            direction_decimals (int): The desired number of decimal places for
                the normal. The default is `DEFAULT_DIRECTION_DECIMALS`.

        Returns:
            dict: The JSON representation.
        """
        rounded = self.rounded(
            position_decimals=position_decimals, direction_decimals=direction_decimals
        )
        return {
            "referencePoint": rounded.reference_point.tolist(),
            "unitNormal": rounded.normal.tolist(),
        }

    @classmethod
    def validate(cls, data):
        """
        Validate a plane JSON representation.

        The schema is defined in `types/src/schema.json`.

        Args:
            data (dict): The JSON representation.
        """
        from .._common.pathlib import SCHEMA_PATH
        from .._common.serialization import validator_for

        try:
            validator = cls._validator
        except AttributeError:
            validator = None

        if validator is None:
            validator = cls._validator = validator_for(
                schema_path=SCHEMA_PATH,
                ref="#/definitions/Plane",
            )

        validator.validate(data)

    @classmethod
    def deserialize(cls, data):
        """
        Create a Plane from the given JSON representation.

        The schema is defined in `types/src/schema.json`.

        Args:
            data (dict): The JSON representation.

        Returns:
            Plane: The deserialized plane.
        """
        cls.validate(data)

        return cls(
            reference_point=np.array(data["referencePoint"]),
            normal=np.array(data["unitNormal"]),
        )

    @property
    def equation(self):
        """
        Returns parameters `A`, `B`, `C`, `D` as a 1x4 `np.array`, where

            `Ax + By + Cz + D = 0`

        defines the plane.
        """
        A, B, C = self.normal
        D = -self.reference_point.dot(self.normal)

        return np.array([A, B, C, D])

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
        return self.reference_point.dot(self.normal) * self.normal

    def flipped(self):
        """
        Creates a new Plane with an inverted orientation.
        """
        return Plane(reference_point=self.reference_point, normal=-self.normal)

    def flipped_if(self, condition):
        """
        Conditionally flip the plane, returning `self` or a new Plane with an
        inverted orientation.
        """
        return self.flipped() if condition else self

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
        return Plane(reference_point=coplanar_point, normal=new_normal)


Plane.xy = Plane(reference_point=np.zeros(3), normal=vg.basis.z)
Plane.xy.__doc__ = "The `xy`-plane."
Plane.xz = Plane(reference_point=np.zeros(3), normal=vg.basis.y)
Plane.xz.__doc__ = "The `xz`-plane."
Plane.yz = Plane(reference_point=np.zeros(3), normal=vg.basis.x)
Plane.yz.__doc__ = "The `yz`-plane."
