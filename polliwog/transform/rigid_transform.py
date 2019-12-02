import vg


def find_rigid_transform(a, b, compute_scale=False, fail_in_degenerate_cases=True):
    """
    Args:
        a: a Nx3 array of vertex locations
        b: a Nx3 array of vertex locations
        a and b are in correspondence -- we find a transformation such that the first
        point in a will be moved to the location of the first point in b, etc.

    Returns: (R,T) such that a.dot(R) + T ~= b
        R is a 3x3 rotation matrix
        T is a 1x3 translation vector

    Based on Arun et al, "Least-squares fitting of two 3-D point sets," 1987.
    See also Eggert et al, "Estimating 3-D rigid body transformations: a
    comparison of four major algorithms," 1997.

    If compute_scale is True, also computes and returns: (s, R,T) such that s*(R.dot(a))+T ~= b

    In noisy cases, when there is a reflection, this algorithm can fail. In those cases the
    right thing to do is to try a less noise sensitive algorithm like RANSAC. But if you want
    a result anyway, even knowing that it might not be right, set fail_in_degenerate_cases=True.

    """
    import numpy as np

    k = vg.shape.check(locals(), "a", (-1, 3))
    vg.shape.check(locals(), "b", (k, 3))

    a = a.T
    b = b.T

    a_mean = np.mean(a, axis=1)
    b_mean = np.mean(b, axis=1)
    a_centered = a - a_mean.reshape(-1, 1)
    b_centered = b - b_mean.reshape(-1, 1)

    c = a_centered.dot(b_centered.T)
    u, s, v = np.linalg.svd(c, full_matrices=False)
    v = v.T
    R = v.dot(u.T)

    if np.linalg.det(R) < 0:
        if (
            np.any(s == 0) or not fail_in_degenerate_cases
        ):  # This is only valid in the noiseless case; see the paper
            v[:, 2] = -v[:, 2]
            R = v.dot(u.T)
        else:
            raise ValueError(
                "find_rigid_transform found a reflection that it cannot recover from. Try RANSAC or something..."
            )

    if compute_scale:
        scale = np.sum(s) / (np.linalg.norm(a_centered) ** 2)
        T = (b_mean - scale * (R.dot(a_mean))).reshape(1, 3)
        return scale, R.T, T
    else:
        T = (b_mean - R.dot(a_mean)).reshape(1, 3)
        return R.T, T


def find_rigid_rotation(a, b, allow_scaling=False):
    """
    Args:
        a: a Nx3 array of vertex locations
        b: a Nx3 array of vertex locations

    Returns: R such that a.dot(R) ~= b

    See link: http://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """
    import numpy as np

    k = vg.shape.check(locals(), "a", (-1, 3))
    vg.shape.check(locals(), "b", (k, 3))

    a = a.T
    b = b.T

    if a.size == 3:
        cx = np.cross(a.ravel(), b.ravel())
        a = np.hstack([a.reshape(-1, 1), cx.reshape(-1, 1)])
        b = np.hstack([b.reshape(-1, 1), cx.reshape(-1, 1)])

    c = a.dot(b.T)
    u, _, v = np.linalg.svd(c, full_matrices=False)
    v = v.T
    R = v.dot(u.T)

    if np.linalg.det(R) < 0:
        v[:, 2] = -v[:, 2]
        R = v.dot(u.T)

    if allow_scaling:
        scalefactor = np.linalg.norm(b) / np.linalg.norm(a)
        R = R * scalefactor

    return R.T
