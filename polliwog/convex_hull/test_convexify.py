import unittest
import numpy as np
from blmath.geometry import Polyline
from blmath.geometry.convexify import convexify_planar_curve

class TestConvexify(unittest.TestCase):

    def test_convexify_degenerate_cases(self):
        # Sometimes a cross-sectional slice lands right on a vertex, e.g. the
        # horizontal cross section passing through the highest point on the
        # head. Such a slice is already convex.
        collapsed_polyline = Polyline(np.array([1., 2., 3.]).reshape((-1, 3)))

        self.assertEqual(
            convexify_planar_curve(collapsed_polyline),
            collapsed_polyline
        )

        # Are no points convex? The alternative would be to raise an
        # exception...
        empty_polyline = Polyline(np.array([]).reshape((-1, 3)))

        self.assertEqual(
            convexify_planar_curve(empty_polyline),
            empty_polyline
        )
