import numpy as np
import vg
from ._plane_intersect import intersect_segment_with_plane


def test_intersect_segment_with_plane_single():
    start_point = np.array([0.0, 9.0, 0.0])
    segment_vector = np.array([0.0, -9.0, 0.0])

    np.testing.assert_array_equal(
        intersect_segment_with_plane(
            start_points=start_point,
            segment_vectors=segment_vector,
            points_on_plane=np.array([0.0, 7.5, 0.0]),
            plane_normals=vg.basis.y,
        ),
        np.array([0.0, 7.5, 0.0]),
    )


def test_intersect_segment_with_plane_stacked():
    start_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 7.0, 0.0],
            [1.0, 9.0, 0.0],
            [0.0, 9.0, 0.0],
        ]
    )
    segment_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 2.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -9.0, 0.0],
        ]
    )

    num_points = len(start_points)

    np.testing.assert_array_equal(
        intersect_segment_with_plane(
            start_points=start_points,
            segment_vectors=segment_vectors,
            points_on_plane=np.broadcast_to(np.array([0.0, 7.5, 0.0]), (num_points, 3)),
            plane_normals=np.broadcast_to(vg.basis.y, (num_points, 3)),
        ),
        np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [1.0, 7.5, 0.0],
                [np.nan, np.nan, np.nan],
                [0.0, 7.5, 0.0],
            ]
        ),
    )
