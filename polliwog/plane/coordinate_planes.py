import numpy as np
import vg
from .plane import Plane


class _CoordinatePlanes(object):
    """
    The planes  basis vectors.
    """

    @property
    def xy(self):
        return Plane(point_on_plane=np.zeros(3), unit_normal=vg.basis.z)

    @property
    def xz(self):
        return Plane(point_on_plane=np.zeros(3), unit_normal=vg.basis.y)

    @property
    def yz(self):
        return Plane(point_on_plane=np.zeros(3), unit_normal=vg.basis.x)


coordinate_planes = _CoordinatePlanes()
