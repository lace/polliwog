import numpy as np
import vg


def intersect_segment_with_plane(
    start_points, segment_vectors, points_on_plane, plane_normals
):
    k = vg.shape.check(locals(), "start_points", (-1, 3))
    vg.shape.check(locals(), "segment_vectors", (k, 3))
    vg.shape.check(locals(), "points_on_plane", (k, 3))
    vg.shape.check(locals(), "plane_normals", (k, 3))

    # Compute t values such that
    # `result = reference_point + t * segment_vectors`.
    t = np.nan_to_num(
        vg.dot(points_on_plane - start_points, plane_normals)
        / vg.dot(segment_vectors, plane_normals)
    )

    intersection_points = start_points + t.reshape(-1, 1) * segment_vectors

    # Discard points which lie past the ends of the segment.
    intersection_points[t < 0] = np.nan
    intersection_points[t > 1] = np.nan

    return intersection_points
