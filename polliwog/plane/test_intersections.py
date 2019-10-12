import numpy as np
import vg
from .intersections import intersect_segment_with_plane


def test_intersect_plane_new():
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

    from ..polyline.polyline import Polyline

    polyline = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 9.0, 0.0],
                [0.0, 9.0, 0.0],
            ]
        ),
        closed=True,
    )
    np.testing.assert_array_equal(start_points, polyline.segments[:, 0])
    np.testing.assert_array_equal(segment_vectors, polyline.segment_vectors)

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
