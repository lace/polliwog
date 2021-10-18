import numpy as np
from vg.compat import v2 as vg
from .._common.shape import check_shape_any, columnize
from ..line._line_functions import coplanar_points_are_on_same_side_of_line

FACE_DTYPE = np.int64

__all__ = [
    "FACE_DTYPE",
    "edges_of_faces",
    "surface_normals",
    "surface_area",
    "tri_contains_coplanar_point",
    "barycentric_coordinates_of_points",
    "sample",
]


def edges_of_faces(faces, normalize=True):
    """
    Given a stack of triangles expressed as vertex indices, return a
    normalized array of edges. When `normalize=True`, sort the edges so they
    more easily can be compared.
    """
    vg.shape.check(locals(), "faces", (-1, 3))
    assert faces.dtype == FACE_DTYPE

    # TODO: It's probably possible to accomplish this more efficiently. Maybe
    # with `np.pick()`?
    interleaved_edges = np.stack(
        [faces[:, 0:2], faces[:, 1:3], np.roll(faces, 1, axis=1)[:, 0:2]]
    )
    flattened_edges = np.swapaxes(interleaved_edges, 0, 1).reshape(-1, 2)
    return np.sort(flattened_edges, axis=1) if normalize else flattened_edges


def surface_normals(points, normalize=True):
    """
    Compute the surface normal of a triangle. The direction of the normal
    follows conventional counter-clockwise winding and the right-hand
    rule.

    Also works on stacked inputs (i.e. many sets of three points).
    """
    points, _, transform_result = columnize(points, (-1, 3, 3), name="points")

    p1s = points[:, 0]
    p2s = points[:, 1]
    p3s = points[:, 2]
    v1s = p2s - p1s
    v2s = p3s - p1s
    normals = vg.cross(v1s, v2s)

    if normalize:
        normals = vg.normalize(normals)

    return transform_result(normals)


def surface_area(vertices_of_tris):
    """
    Compute the surface area of a triangle or triangles.

    Args:
        vertices_of_tris (np.arraylike): A set of triangle vertices as `3x3` or
        `kx3x3`.

    Returns:
        object: For `3x3` input, a float containing the area. For `kx3x3` input,
        an array containing the area of each triangle.
    """
    points, _, transform_result = columnize(
        vertices_of_tris, (-1, 3, 3), name="vertices_of_tris"
    )

    e1s = points[:, 1] - points[:, 0]
    e2s = points[:, 2] - points[:, 0]

    cross_products = np.array(
        [
            e1s[:, 1] * e2s[:, 2] - e1s[:, 2] * e2s[:, 1],
            e1s[:, 2] * e2s[:, 0] - e1s[:, 0] * e2s[:, 2],
            e1s[:, 0] * e2s[:, 1] - e1s[:, 1] * e2s[:, 0],
        ]
    ).T
    areas = 0.5 * np.sqrt((cross_products ** 2).sum(axis=1))

    return transform_result(areas)


def tri_contains_coplanar_point(a, b, c, point):
    """
    Assuming `point` is coplanar with the triangle `ABC`, check if it lies
    inside it.
    """
    check_shape_any(a, (3,), (-1, 3), name="a")
    vg.shape.check(locals(), "b", a.shape)
    vg.shape.check(locals(), "c", a.shape)
    vg.shape.check(locals(), "point", a.shape)

    # Uses "same-side technique" from http://blackpawn.com/texts/pointinpoly/default.html
    return np.logical_and(
        np.logical_and(
            coplanar_points_are_on_same_side_of_line(b, c, point, a),
            coplanar_points_are_on_same_side_of_line(a, c, point, b),
        ),
        coplanar_points_are_on_same_side_of_line(a, b, point, c),
    )


def barycentric_coordinates_of_points(vertices_of_tris, points):
    """
    Compute barycentric coordinates for the projection of a set of points to a
    given set of triangles specfied by their vertices.

    These barycentric coordinates can refer to points outside the triangle.
    This happens when one of the coordinates is negative. However they can't
    specify points outside the triangle's plane. (That requires tetrahedral
    coordinates.)

    The returned coordinates supply a linear combination which, applied to the
    vertices, returns the projection of the original point the plane of the
    triangle.

    Args:
        vertices_of_tris (np.arraylike): A set of triangle vertices as `kx3x3`.
        points (np.arraylike): Coordinates of points as `kx3`.

    Returns:
        np.ndarray: Barycentric coordinates as `kx3`

    See Also:
        - https://en.wikipedia.org/wiki/Barycentric_coordinate_system
        - Heidrich, "Computing the Barycentric Coordinates of a Projected
          Point," JGT 05 (http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf)
    """
    k = vg.shape.check(locals(), "vertices_of_tris", (-1, 3, 3))
    vg.shape.check(locals(), "points", (k, 3))

    p = points.T
    q = vertices_of_tris[:, 0].T
    u = (vertices_of_tris[:, 1] - vertices_of_tris[:, 0]).T
    v = (vertices_of_tris[:, 2] - vertices_of_tris[:, 0]).T

    n = np.cross(u, v, axis=0)
    s = np.sum(n * n, axis=0)

    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = np.spacing(1)

    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = np.sum(np.cross(u, w, axis=0) * n, axis=0) * oneOver4ASquared
    b1 = np.sum(np.cross(w, v, axis=0) * n, axis=0) * oneOver4ASquared
    b = np.vstack((1 - b1 - b2, b1, b2))

    return b.T


RANDOM_SEED = 1337


def sample(
    vertices_of_tris,
    num_samples,
    rng=None,
    weights=None,
    ret_points=True,
    ret_face_indices=False,
):
    """
    Generate points sampled across the surface of the triangles provided.

    By default, triangles are weighted by area, and the random number seed is
    fixed, making this function deterministic.

    Args:
        vertices_of_tris (np.arraylike): A set of triangle vertices as `kx3x3`.
        num_samples (int): The number of samples desired.
        rng (np.random.Generator): A NumPy random number generator. To obtain
            random results for each invocation, pass `np.random.default_rng()`.
        weights (np.arraylike): `kx3` weights. By default, triangles are
            weighted by area.
        ret_face_indices (bool): When True, return the face indices from
            which each point was taken.

    Returns:
        object: The sampledpoints, or a tuple containing the sampled points and
        their corresponding face indices.
    """
    k = vg.shape.check(locals(), "vertices_of_tris", (-1, 3, 3))
    if not isinstance(num_samples, int):
        raise ValueError("Expected num_samples to be an int")
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    elif not isinstance(rng, np.random.Generator):
        raise ValueError("Expected rng to be an instance of np.random.Generator")
    if weights is None:
        weights = surface_area(vertices_of_tris)
    else:
        vg.shape.check(locals(), "weights", (k,))

    if k == 0:
        empty_samples = np.zeros((0, 3), dtype=np.float64)
        if ret_face_indices:
            empty_face_indices = np.zeros((0))
            return empty_samples, empty_face_indices
        else:
            return empty_samples

    # Adapted from Trimesh, Copyright (c) 2019 Michael Dawson-Haggerty
    # https://github.com/mikedh/trimesh/blob/00ffb55b150d37988f6113a758c50f4092a49bd1/trimesh/sample.py
    # and
    # https://blogs.sas.com/content/iml/2020/10/19/random-points-in-triangle.html

    # Pick which triangles will be sampled.
    cumulative_weights = np.cumsum(weights)
    total_weight = cumulative_weights[-1]
    face_indices = np.searchsorted(
        cumulative_weights, rng.random(num_samples) * total_weight
    )

    v0s = vertices_of_tris[face_indices, 0]
    edge_vectors = vertices_of_tris[face_indices, 1:] - np.tile(v0s, (1, 2)).reshape(
        (-1, 2, 3)
    )

    # Generate scalar coefficients from 0-1 for each edge vector. When
    # multiplied out, this produces samples of parallelograms, of which the
    # triangle makes up half.  When the sum of the coefficients `u` and `v` is >
    # 1, this point will be outside the triangle. Transform `u` to `1 - u` and
    # `v` to `1 - v` to reflect the sample across the last edge, back onto the
    # triangle.
    coeffs = rng.random((num_samples, 2, 1))
    coeffs_needing_reflection = coeffs.sum(axis=1).ravel() > 1
    coeffs[coeffs_needing_reflection] = 1 - coeffs[coeffs_needing_reflection]

    # Multiply out the vectors to produce the samples.
    samples = v0s + (coeffs * edge_vectors).sum(axis=1)

    return (samples, face_indices) if ret_face_indices else samples
