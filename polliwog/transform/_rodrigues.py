import numpy as np
from vg.compat import v2 as vg


def rodrigues_vector_to_rotation_matrix(r, calculate_jacobian=False):
    """
    Convert a 3x1 or 1x3 Rodrigues vector to a 3x3 rotation matrix.

    A Rodrigues vector is a 3 element vector representing a 3D rotation.
    Its direction represents the axis about which to rotate and its magnitude
    represents the amount to rotate by.

    All of SO3 (that is, all 3D rotations) can be uniquely represented by a
    Rodrigues vector, and it does not suffer from the multiple representation
    and gimbal locking problems that Euler angle representations do.

    If `calculate_jacobian` is passed, then the derivative of the rotation is
    also computed. Note that the derivative is undefined for a Rodrigues vector
    of `[0,0,0]` (that is, no rotation).

    See also:

        - https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    r = np.array(r, dtype=np.double)
    r = r.flatten()
    vg.shape.check_value(r, (3,))
    eps = np.finfo(np.double).eps
    theta = np.linalg.norm(r)
    if theta < eps:
        r_out = np.eye(3)
        if calculate_jacobian:
            jac = np.zeros((3, 9))
            jac[0, 5] = jac[1, 6] = jac[2, 1] = -1
            jac[0, 7] = jac[1, 2] = jac[2, 3] = 1
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        c1 = 1.0 - c
        itheta = 1.0 if theta == 0.0 else 1.0 / theta
        r *= itheta
        I = np.eye(3)  # noqa: E741 I is an identity matrix
        rrt = np.array([r * r[0], r * r[1], r * r[2]])
        _r_x_ = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        r_out = c * I + c1 * rrt + s * _r_x_
        if calculate_jacobian:
            drrt = np.array(
                [
                    [r[0] + r[0], r[1], r[2], r[1], 0, 0, r[2], 0, 0],
                    [0, r[0], 0, r[0], r[1] + r[1], r[2], 0, r[2], 0],
                    [0, 0, r[0], 0, 0, r[1], r[0], r[1], r[2] + r[2]],
                ]
            )
            d_r_x_ = np.array(
                [
                    [0, 0, 0, 0, 0, -1, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, -1, 0, 0],
                    [0, -1, 0, 1, 0, 0, 0, 0, 0],
                ]
            )
            I_jac = np.array([I.flatten(), I.flatten(), I.flatten()])
            ri = np.array([[r[0]], [r[1]], [r[2]]])
            a0 = -s * ri
            a1 = (s - 2 * c1 * itheta) * ri
            a2 = np.ones((3, 1)) * c1 * itheta
            a3 = (c - s * itheta) * ri
            a4 = np.ones((3, 1)) * s * itheta
            jac = (
                a0 * I_jac
                + a1 * rrt.flatten()
                + a2 * drrt
                + a3 * _r_x_.flatten()
                + a4 * d_r_x_
            )
    if calculate_jacobian:
        return r_out, jac
    else:
        return r_out


def rotation_matrix_to_rodrigues_vector(r, calculate_jacobian=False):
    """
    Convert a 3x3 rotation matrix to a 3x1 or 1x3 Rodrigues vector.

    A Rodrigues vector is a 3 element vector representing a 3D rotation.
    Its direction represents the axis about which to rotate and its magnitude
    represents the amount to rotate by.

    All of SO3 (that is, all 3D rotations) can be uniquely represented by a
    Rodrigues vector, and it does not suffer from the multiple representation
    and gimbal locking problems that Euler angle representations do.

    If `calculate_jacobian` is passed, then the derivative of the rotation is
    also computed. Note that the derivative is undefined for a Rodrigues vector
    of `[0,0,0]` (that is, no rotation).

    See also:

        - https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    r = np.array(r, dtype=np.double)
    vg.shape.check_value(r, (3, 3))
    u, _, v = np.linalg.svd(r)
    r = np.dot(u, v)
    rx = r[2, 1] - r[1, 2]
    ry = r[0, 2] - r[2, 0]
    rz = r[1, 0] - r[0, 1]
    s = np.linalg.norm(np.array([rx, ry, rz])) * np.sqrt(0.25)
    c = np.clip((np.sum(np.diag(r)) - 1) * 0.5, -1, 1)
    theta = np.arccos(c)
    if s < 1e-5:
        if c > 0:
            r_out = np.zeros((3, 1))
        else:
            rx, ry, rz = np.clip(np.sqrt((np.diag(r) + 1) * 0.5), 0, np.inf)
            if r[0, 1] < 0:
                ry = -ry
            if r[0, 2] < 0:
                rz = -rz
            if (
                np.abs(rx) < np.abs(ry)
                and np.abs(rx) < np.abs(rz)
                and ((r[1, 2] > 0) != (ry * rz > 0))
            ):
                rz = -rz
            r_out = np.array([[rx, ry, rz]]).T
            theta /= np.linalg.norm(r_out)
            r_out *= theta
        if calculate_jacobian:
            jac = np.zeros((9, 3))
            if c > 0:
                jac[1, 2] = jac[5, 0] = jac[6, 1] = -0.5
                jac[2, 1] = jac[3, 2] = jac[7, 0] = 0.5
    else:
        vth = 1.0 / (2.0 * s)
        if calculate_jacobian:
            dtheta_dtr = -1.0 / s
            dvth_dtheta = -vth * c / s
            d1 = 0.5 * dvth_dtheta * dtheta_dtr
            d2 = 0.5 * dtheta_dtr
            dvardR = np.array(
                [
                    [0, 0, 0, 0, 0, 1, 0, -1, 0],
                    [0, 0, -1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, -1, 0, 0, 0, 0, 0],
                    [d1, 0, 0, 0, d1, 0, 0, 0, d1],
                    [d2, 0, 0, 0, d2, 0, 0, 0, d2],
                ]
            )
            dvar2dvar = np.array(
                [
                    [vth, 0, 0, rx, 0],
                    [0, vth, 0, ry, 0],
                    [0, 0, vth, rz, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
            domegadvar2 = np.array(
                [
                    [theta, 0, 0, rx * vth],
                    [0, theta, 0, ry * vth],
                    [0, 0, theta, rz * vth],
                ]
            )
            jac = np.dot(np.dot(domegadvar2, dvar2dvar), dvardR)
            for ii in range(3):
                jac[ii] = jac[ii].reshape((3, 3)).T.flatten()
            jac = jac.T
        vth *= theta
        r_out = np.array([[rx, ry, rz]]).T * vth
    if calculate_jacobian:
        return r_out, jac
    else:
        return r_out


def cv2_rodrigues(r, calculate_jacobian=False):
    """
    `cv2_rodrigues` is a wrapped function designed to be API compatible with
    OpenCV's `cv2.Rodrigues`.

    If it is given a rotation matrix, it returns a Rodrigues vector.

    If it is given a Rodrigues vector, it returns a rotation matrix.

    To make your code clearer, call `rodrigues_vector_to_rotation_matrix` or
    `rotation_matrix_to_rodrigues_vector` directly, which makes the intent of
    your code clearer.
    """
    r = np.array(r, dtype=np.double)
    if r.size == 3:
        return rodrigues_vector_to_rotation_matrix(r, calculate_jacobian)
    elif r.shape == (3, 3):
        return rotation_matrix_to_rodrigues_vector(r, calculate_jacobian)
    else:
        raise ValueError("rodrigues: input matrix must be 1x3, 3x1 or 3x3.")
