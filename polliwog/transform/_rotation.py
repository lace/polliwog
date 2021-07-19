import numpy as np
from vg.compat import v2 as vg


def rotation_from_up_and_look(up, look):
    """
    Rotation matrix to rotate a mesh into a canonical reference frame. The
    result is a rotation matrix that will make up along +y and look along +z
    (i.e. facing towards a default opengl camera).

    up: The direction you want to become `+y`.
    look: The direction you want to become `+z`.

    """
    vg.shape.check(locals(), "up", (3,))
    vg.shape.check(locals(), "look", (3,))

    up, look = [np.asarray(vector, dtype=np.float64) for vector in (up, look)]

    if np.linalg.norm(up) == 0:
        raise ValueError("Singular up")
    if np.linalg.norm(look) == 0:
        raise ValueError("Singular look")

    y = up / np.linalg.norm(up)
    z = look - np.dot(look, y) * y
    if np.linalg.norm(z) == 0:
        raise ValueError("up and look are collinear")
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    return np.array([x, y, z])


def euler(xyz, order="xyz", units="deg"):
    """
    Convert a Euler angle representation of 3D rotations to a 3x3 rotation matrix.

    Euler angles are a way of representing 3D rotations as a sequence of rotations
    about the axes. Conceptually, think of `euler([10, 20, 30])` as
    "Rotate 10 degrees around the x axis, then 20 degrees around the y axis, then
    30 degrees around the z axis" (that ordering can be changed with the `order`
    argument, and the units can be given in degrees or radians by setting `units`
    to `'deg'` or `'rad'`).

    Euler angles are a problematic representation of rotation for numerical methods,
    as there are multiple possible representations for a given rotation. But they are
    a very intuitive and readable way to initialize a rotation matrix.

    See also:

        - https://en.wikipedia.org/wiki/Euler_angles
    """
    if not hasattr(xyz, "__iter__"):
        xyz = [xyz]
    if units == "deg":
        xyz = np.radians(xyz)
    r = np.eye(3)
    for theta, axis in zip(xyz, order):
        c = np.cos(theta)
        s = np.sin(theta)
        if axis == "x":
            r = np.dot(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), r)
        if axis == "y":
            r = np.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), r)
        if axis == "z":
            r = np.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), r)
    return r
