import numpy as np
import vg
from .coordinate_planes import coordinate_planes


def test_constants():
    np.testing.assert_array_equal(coordinate_planes.xy.normal, vg.basis.z)
    np.testing.assert_array_equal(coordinate_planes.xz.normal, vg.basis.y)
    np.testing.assert_array_equal(coordinate_planes.yz.normal, vg.basis.x)
