def find_rigid_transform(a, b, visualize=False):
    """
    Args:
        a: a 3xN array of vertex locations
        b: a 3xN array of vertex locations

    Returns: (R,T) such that R.dot(a)+T ~= b
    Based on Arun et al, "Least-squares fitting of two 3-D point sets," 1987.
    See also Eggert et al, "Estimating 3-D rigid body transformations: a
    comparison of four major algorithms," 1997.
    """
    import numpy as np
    import scipy.linalg
    from blmath.numerics.matlab import col

    if a.shape[0] != 3:
        if a.shape[1] == 3:
            a = a.T
    if b.shape[0] != 3:
        if b.shape[1] == 3:
            b = b.T
    assert a.shape[0] == 3
    assert b.shape[0] == 3

    a_mean = np.mean(a, axis=1)
    b_mean = np.mean(b, axis=1)
    a_centered = a - col(a_mean)
    b_centered = b - col(b_mean)

    c = a_centered.dot(b_centered.T)
    u, s, v = np.linalg.svd(c, full_matrices=False)
    v = v.T
    R = v.dot(u.T)

    if scipy.linalg.det(R) < 0:
        if np.any(s == 0): # This is only valid in the noiseless case; see the paper
            v[:, 2] = -v[:, 2]
            R = v.dot(u.T)
        else:
            raise ValueError("find_rigid_transform found a reflection that it cannot recover from. Try RANSAC or something...")

    T = col(b_mean - R.dot(a_mean))

    if visualize != False:
        from lace.mesh import Mesh
        from lace.meshviewer import MeshViewer
        mv = MeshViewer() if visualize is True else visualize
        a_T = R.dot(a) + T
        mv.set_dynamic_meshes([
            Mesh(v=a.T, f=[]).set_vertex_colors('red'),
            Mesh(v=b.T, f=[]).set_vertex_colors('green'),
            Mesh(v=a_T.T, f=[]).set_vertex_colors('orange'),
        ])

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
    import scipy.linalg
    from blmath.numerics.matlab import col

    assert a.shape[0] == 3
    assert b.shape[0] == 3

    if a.size == 3:
        cx = np.cross(a.ravel(), b.ravel())
        a = np.hstack((col(a), col(cx)))
        b = np.hstack((col(b), col(cx)))

    c = a.dot(b.T)
    u, _, v = np.linalg.svd(c, full_matrices=False)
    v = v.T
    R = v.dot(u.T)

    if scipy.linalg.det(R) < 0:
        v[:, 2] = -v[:, 2]
        R = v.dot(u.T)

    if allow_scaling:
        scalefactor = scipy.linalg.norm(b) / scipy.linalg.norm(a)
        R = R * scalefactor

    return R
