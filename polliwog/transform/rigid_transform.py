import vg


def find_rigid_transform(a, b, compute_scale=False, fail_in_degenerate_cases=True):
    """
    Args:
        a: a 3xN array of vertex locations
        b: a 3xN array of vertex locations
        a and b are in correspondence -- we find a transformation such that the first
        point in a will be moved to the location of the first point in b, etc.

    Returns: (R,T) such that R.dot(a)+T ~= b
        R is a 3x3 rotation matrix
        T is a 3x1 translation vector

    Based on Arun et al, "Least-squares fitting of two 3-D point sets," 1987.
    See also Eggert et al, "Estimating 3-D rigid body transformations: a
    comparison of four major algorithms," 1997.

    If compute_scale is True, also computes and returns: (s, R,T) such that s*(R.dot(a))+T ~= b

    In noisy cases, when there is a reflection, this algorithm can fail. In those cases the
    right thing to do is to try a less noise sensitive algorithm like RANSAC. But if you want
    a result anyway, even knowing that it might not be right, set fail_in_degenerate_cases=True.

    """
    import numpy as np

    vg.shape.check(locals(), "a", (3, -1))
    vg.shape.check(locals(), "b", (3, -1))

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
        T = (b_mean - scale * (R.dot(a_mean))).reshape(-1, 1)
        return scale, R, T
    else:
        T = (b_mean - R.dot(a_mean)).reshape(-1, 1)
        return R, T


def find_rigid_rotation(a, b, allow_scaling=False):
    """
    Args:
        a: a 3xN array of vertex locations
        b: a 3xN array of vertex locations

    Returns: R such that R.dot(a) ~= b

    See link: http://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """
    import numpy as np

    assert a.shape[0] == 3
    assert b.shape[0] == 3

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

    return R
