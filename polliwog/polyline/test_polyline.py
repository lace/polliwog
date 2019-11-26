import numpy as np
import pytest
import vg
from .polyline import Polyline
from ..plane.plane import Plane


def test_join():
    vs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    more_vs = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 3.0, 0.0],
        ]
    )
    joined = Polyline.join(
        Polyline(vs, is_closed=False),
        Polyline(more_vs, is_closed=False),
        is_closed=True,
    )
    assert joined.is_closed is True
    np.testing.assert_array_equal(joined.v, np.vstack([vs, more_vs]))

    with pytest.raises(ValueError, match="Need at least one polyline to join"):
        Polyline.join()

    with pytest.raises(
        ValueError, match="Expected input polylines to be open, not closed"
    ):
        Polyline.join(Polyline(vs, is_closed=False), Polyline(more_vs, is_closed=True))


def test_repr():
    example_vs = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    assert (
        repr(Polyline(example_vs, is_closed=True)) == "<closed Polyline with 4 verts>"
    )
    assert repr(Polyline(example_vs, is_closed=False)) == "<open Polyline with 4 verts>"
    assert (
        repr(Polyline(np.array([]).reshape(-1, 3), is_closed=False))
        == "<Polyline with no verts>"
    )


def test_to_dict():
    example_vs = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    expected_dict = {
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
    }
    actual_dict = Polyline(example_vs, is_closed=True).to_dict(decimals=3)
    # TODO Is there a cleaner way to assert deep equal?
    assert set(actual_dict.keys()) == set(expected_dict.keys())
    np.testing.assert_array_equal(expected_dict["vertices"], actual_dict["vertices"])
    np.testing.assert_array_equal(expected_dict["edges"], actual_dict["edges"])


def test_bounding_box():
    bounding_box = Polyline(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    ).bounding_box
    np.testing.assert_array_equal(bounding_box.origin, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(bounding_box.size, np.array([1.0, 2.0, 0.0]))


def test_bounding_box_degnerate():
    bounding_box = Polyline(np.array([[0.0, 0.0, 0.0]])).bounding_box
    np.testing.assert_array_equal(bounding_box.origin, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(bounding_box.size, np.array([0.0, 0.0, 0.0]))

    assert Polyline(np.zeros((0, 3))).bounding_box is None


def test_index_of_vertex():
    polyline = Polyline(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    )
    assert polyline.index_of_vertex(np.array([0.0, 0.0, 0.0])) == 0
    assert polyline.index_of_vertex(np.array([1.0, 2.0, 0.0])) == 3
    with pytest.raises(ValueError, match="No matching vertex"):
        polyline.index_of_vertex(np.array([1.0, 2.0, 3.0]))


def test_with_insertions():
    original_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    points_to_insert = np.array([[0.5, 0.0, 0.0], [1.0, 1.5, 0.0]])
    before_indices = np.array([1, 3])
    expected_v = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.5, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    expected_indices_of_original_vertices = np.array([0, 2, 3, 5])
    expected_indices_of_inserted_points = np.array([1, 4])

    polyline = Polyline(v=original_points, is_closed=True,).with_insertions(
        points=points_to_insert, indices=before_indices,
    )
    assert polyline.is_closed is True
    np.testing.assert_array_almost_equal(polyline.v, expected_v)

    polyline, indices_of_original_vertices, indices_of_inserted_points = Polyline(
        v=original_points, is_closed=False,
    ).with_insertions(
        points=points_to_insert, indices=before_indices, ret_new_indices=True,
    )
    assert polyline.is_closed is False
    np.testing.assert_array_almost_equal(polyline.v, expected_v)
    np.testing.assert_array_equal(
        indices_of_original_vertices, expected_indices_of_original_vertices
    )
    np.testing.assert_array_equal(
        indices_of_inserted_points, expected_indices_of_inserted_points
    )

    polyline, indices_of_original_vertices, indices_of_inserted_points = Polyline(
        v=original_points, is_closed=False,
    ).with_insertions(
        points=np.flipud(points_to_insert),
        indices=np.flip(before_indices),
        ret_new_indices=True,
    )
    assert polyline.is_closed is False
    np.testing.assert_array_almost_equal(polyline.v, expected_v)
    np.testing.assert_array_equal(
        indices_of_original_vertices, expected_indices_of_original_vertices
    )
    np.testing.assert_array_equal(
        indices_of_inserted_points, np.flip(expected_indices_of_inserted_points)
    )


def test_update_is_closed():
    example_vs = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    polyline = Polyline(example_vs, is_closed=False)
    assert polyline.num_e == 3
    assert polyline.is_closed is False
    polyline.is_closed = True
    assert polyline.num_e == 4
    assert polyline.is_closed is True


def test_num_v_num_e():
    example_vs = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    closed_polyline = Polyline(example_vs, is_closed=True)
    assert closed_polyline.num_v == 4
    assert len(closed_polyline) == 4
    assert closed_polyline.num_e == 4
    open_polyline = Polyline(example_vs, is_closed=False)
    assert open_polyline.num_v == 4
    assert open_polyline.num_e == 3


def test_edges():
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 3.0, 0.0],
        ]
    )

    expected_open = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])

    np.testing.assert_array_equal(Polyline(v).e, expected_open)

    expected_closed = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])

    np.testing.assert_array_equal(Polyline(v, is_closed=True).e, expected_closed)


def test_segments():
    polyline = Polyline(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]),
        is_closed=True,
    )

    expected = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
            [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    np.testing.assert_array_equal(polyline.segments, expected)


def test_segment_vectors():
    polyline = Polyline(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]),
        is_closed=True,
    )

    expected = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -2.0, 0.0]]
    )

    np.testing.assert_array_equal(polyline.segment_vectors, expected)


def test_length_of_empty_polyline():
    polyline = Polyline(None)
    assert polyline.total_length == 0

    polyline = Polyline(None, is_closed=True)
    assert polyline.total_length == 0


def test_partition_by_length_noop():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        )
    )

    result = original.copy()
    indices = result.partition_by_length(1.0, ret_indices=True)

    expected_indices = np.array([0, 1, 2, 3, 4])

    np.testing.assert_array_almost_equal(result.v, original.v)
    np.testing.assert_array_equal(result.e, original.e)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_partition_by_length_degenerate():
    """
    This covers a bug that arose from a numerical stability issue in
    measurement on EC2 / MKL.
    """
    original = Polyline(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

    result = original.copy()
    indices = result.partition_by_length(1.0, ret_indices=True)

    expected_indices = np.array([0, 1, 2])

    np.testing.assert_array_almost_equal(result.v, original.v)
    np.testing.assert_array_equal(result.e, original.e)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_partition_by_length_divide_by_two():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        )
    )

    expected = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.5, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 2.5, 0.0],
                [1.0, 3.0, 0.0],
            ]
        )
    )

    expected_indices = np.array([0, 2, 4, 6, 8])

    for max_length in (0.99, 0.75, 0.5):
        result = original.copy()
        indices = result.partition_by_length(max_length, ret_indices=True)

        np.testing.assert_array_almost_equal(result.v, expected.v)
        np.testing.assert_array_equal(result.e, expected.e)
        np.testing.assert_array_equal(indices, expected_indices)

        result_2 = original.copy()
        ret = result_2.partition_by_length(max_length, ret_indices=False)
        np.testing.assert_array_almost_equal(result.v, expected.v)
        assert ret is result_2
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_partition_length_divide_by_five():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        )
    )

    expected = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.2, 0.0],
                [1.0, 0.4, 0.0],
                [1.0, 0.6, 0.0],
                [1.0, 0.8, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.2, 0.0],
                [1.0, 1.4, 0.0],
                [1.0, 1.6, 0.0],
                [1.0, 1.8, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 2.2, 0.0],
                [1.0, 2.4, 0.0],
                [1.0, 2.6, 0.0],
                [1.0, 2.8, 0.0],
                [1.0, 3.0, 0.0],
            ]
        )
    )

    expected_indices = np.array([0, 5, 10, 15, 20])

    for max_length in (0.2, 0.24):
        result = original.copy()
        indices = result.partition_by_length(max_length, ret_indices=True)

        np.testing.assert_array_almost_equal(result.v, expected.v)
        np.testing.assert_array_equal(result.e, expected.e)
        np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_partition_by_length_divide_some_leave_some():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
            ]
        )
    )

    expected = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 3.0, 0.0],
                [1.0, 5.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
            ]
        )
    )

    expected_indices = np.array([0, 1, 2, 5, 6])

    for max_length in (2.0, 2.99):
        result = original.copy()
        indices = result.partition_by_length(max_length, ret_indices=True)

        np.testing.assert_array_almost_equal(result.v, expected.v)
        np.testing.assert_array_equal(result.e, expected.e)
        np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_partition_by_length_closed():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        ),
        is_closed=True,
    )

    expected = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 3.0, 0.0],
                [1.0, 5.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
                [0.0, 6.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
        is_closed=True,
    )

    expected_indices = np.array([0, 1, 2, 5, 6, 7])

    for max_length in (2.0, 2.5, 2.6):
        result = original.copy()
        indices = result.partition_by_length(max_length, ret_indices=True)

        np.testing.assert_array_almost_equal(result.v, expected.v)
        np.testing.assert_array_equal(result.e, expected.e)
        np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_with_segments_bisected():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        )
    )

    expected = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.5, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        )
    )

    expected_indices_of_original_vertices = np.array([0, 1, 3, 5, 6])

    (
        with_segments_bisected,
        indices_of_original_vertices,
        indices_of_inserted_points,
    ) = original.with_segments_bisected([1, 2], ret_new_indices=True)

    np.testing.assert_array_almost_equal(with_segments_bisected.v, expected.v)
    np.testing.assert_array_equal(with_segments_bisected.e, expected.e)
    np.testing.assert_array_equal(
        indices_of_original_vertices, expected_indices_of_original_vertices
    )
    np.testing.assert_array_equal(
        with_segments_bisected.v[indices_of_original_vertices], original.v
    )


def test_flipped():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        ),
        is_closed=True,
    )

    expected_v = np.array(
        [
            [0.0, 8.0, 0.0],
            [1.0, 8.0, 0.0],
            [1.0, 7.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    flipped = original.flipped()

    assert flipped is not original
    np.testing.assert_array_almost_equal(flipped.v, expected_v)


def test_aligned_with():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        ),
        is_closed=False,
    )

    assert original.aligned_with(vg.basis.y) is original

    np.testing.assert_array_almost_equal(
        original.aligned_with(vg.basis.neg_y).v, np.flipud(original.v)
    )

    assert original.aligned_with(vg.basis.z) is original
    assert original.aligned_with(vg.basis.neg_z) is original


def test_aligned_with_closed():
    with pytest.raises(ValueError, match=r"Can't align a closed polyline"):
        Polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), is_closed=True
        ).aligned_with(vg.basis.y)


def test_aligned_with_degenerate():
    original = Polyline(np.array([[1.0, 2.0, 3.0]]), is_closed=False)
    assert original.aligned_with(vg.basis.y) is original


def test_reindexed():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        ),
        is_closed=True,
    )

    reindexed, edge_mapping = original.rolled(5, ret_edge_mapping=True)

    expected = Polyline(
        np.array(
            [
                [0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
            ]
        ),
        is_closed=True,
    )

    np.testing.assert_array_almost_equal(reindexed.v, expected.v)
    np.testing.assert_array_equal(original.segments[edge_mapping], reindexed.segments)
    np.testing.assert_array_almost_equal(
        original.rolled(5, ret_edge_mapping=False).v, expected.v
    )

    open_polyline = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        ),
        is_closed=False,
    )
    with pytest.raises(ValueError):
        open_polyline.rolled(5)


def test_intersect_plane():
    polyline = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        ),
        is_closed=True,
    )

    expected = np.array([[1.0, 7.5, 0.0], [0.0, 7.5, 0.0]])
    actual = polyline.intersect_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.y)
    )

    np.testing.assert_array_equal(actual, expected)

    intersection_points, edge_indices = polyline.intersect_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.y),
        ret_edge_indices=True,
    )
    np.testing.assert_array_equal(intersection_points, expected)
    np.testing.assert_array_equal(edge_indices, np.array([3, 5]))


@pytest.mark.xfail
def test_intersect_plane_with_vertex_on_plane():
    # TODO: This isn't working correctly.
    # https://github.com/lace/polliwog/issues/72
    polyline = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
                [0.0, 7.5, 0.0],
            ]
        ),
        is_closed=True,
    )

    expected = np.array([[1.0, 7.5, 0.0], [0.0, 7.5, 0.0]])
    actual = polyline.intersect_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.y)
    )

    np.testing.assert_array_equal(actual, expected)


def test_sliced_by_plane_closed():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        ),
        is_closed=True,
    )

    expected = Polyline(
        np.array([[1.0, 7.5, 0.0], [1.0, 8.0, 0.0], [0.0, 8.0, 0.0], [0.0, 7.5, 0.0]]),
        is_closed=False,
    )
    actual = original.sliced_by_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.y)
    )

    np.testing.assert_array_almost_equal(actual.v, expected.v)
    assert actual.is_closed is False

    expected = Polyline(
        np.array(
            [
                [0.0, 7.5, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 7.5, 0.0],
            ]
        ),
        is_closed=False,
    )
    actual = original.sliced_by_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.neg_y)
    )

    np.testing.assert_array_almost_equal(actual.v, expected.v)
    assert actual.is_closed is False

    zigzag = Polyline(
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [5.0, 2.0, 0.0]]),
        is_closed=True,
    )
    with pytest.raises(
        ValueError, match="Polyline intersects the plane too many times"
    ):
        zigzag.sliced_by_plane(
            Plane(point_on_plane=np.array([2.5, 0.0, 0.0]), unit_normal=vg.basis.x)
        )

    with pytest.raises(
        ValueError, match="Polyline has no vertices in front of the plane"
    ):
        original.sliced_by_plane(
            Plane(point_on_plane=np.array([10.0, 0.0, 0.0]), unit_normal=vg.basis.x)
        )


def test_sliced_by_plane_closed_on_vertex():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        ),
        is_closed=True,
    )
    expected = Polyline(
        np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
                [0.0, 8.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        is_closed=False,
    )
    actual = original.sliced_by_plane(
        Plane(point_on_plane=np.array([0.0, 1.0, 0.0]), unit_normal=vg.basis.y)
    )
    np.testing.assert_array_almost_equal(actual.v, expected.v)
    assert actual.is_closed is False


def test_sliced_by_plane_closed_one_vertex():
    original = Polyline(np.array([[0.0, 0.0, 0.0]]), is_closed=True)
    with pytest.raises(
        ValueError, match="Polyline has no vertices in front of the plane"
    ):
        original.sliced_by_plane(
            Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.y)
        )


def test_sliced_by_plane_open():
    original = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 8.0, 0.0],
            ]
        ),
        is_closed=False,
    )

    expected_vs = np.array([[1.0, 7.5, 0.0], [1.0, 8.0, 0.0]])
    actual = original.sliced_by_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.y)
    )

    np.testing.assert_array_almost_equal(actual.v, expected_vs)
    assert actual.is_closed is False

    expected_vs = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 7.0, 0.0],
            [1.0, 7.5, 0.0],
        ]
    )
    actual = original.sliced_by_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.neg_y)
    )

    np.testing.assert_array_almost_equal(actual.v, expected_vs)
    assert actual.is_closed is False

    with pytest.raises(ValueError):
        original.sliced_by_plane(
            Plane(point_on_plane=np.array([0.0, 15.0, 0.0]), unit_normal=vg.basis.neg_y)
        )

    actual = original.sliced_by_plane(
        Plane(
            point_on_plane=np.array([0.5, 0.0, 0.0]),
            unit_normal=vg.normalize(np.array([1.0, -1.0, 0.0])),
        )
    )
    expected_vs = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
    np.testing.assert_array_almost_equal(actual.v, expected_vs)
    assert actual.is_closed is False


def test_apex():
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 3.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(
        Polyline(v, is_closed=False).apex(vg.basis.y), np.array([1.0, 3.0, 0.0])
    )


def test_sliced_at_indices():
    example_vs = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    np.testing.assert_array_almost_equal(
        Polyline(v=example_vs, is_closed=True).sliced_at_indices(0, 2).v,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_almost_equal(
        Polyline(v=example_vs, is_closed=False).sliced_at_indices(0, 2).v,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_almost_equal(
        Polyline(v=example_vs, is_closed=True).sliced_at_indices(0, 0).v,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]),
    )
    np.testing.assert_array_almost_equal(
        Polyline(v=example_vs, is_closed=True).sliced_at_indices(3, 4).v,
        np.array([[1.0, 2.0, 0.0]]),
    )
    np.testing.assert_array_almost_equal(
        Polyline(v=example_vs, is_closed=True).sliced_at_indices(3, 0).v,
        np.array([[1.0, 2.0, 0.0]]),
    )
    np.testing.assert_array_almost_equal(
        Polyline(v=example_vs, is_closed=True).sliced_at_indices(3, 1).v,
        np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_almost_equal(
        Polyline(v=example_vs, is_closed=True).sliced_at_indices(3, 2).v,
        np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_almost_equal(
        Polyline(v=example_vs, is_closed=True).sliced_at_indices(3, 3).v,
        np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
    )
    with pytest.raises(
        ValueError,
        match=r"For an open polyline, start index of slice should be less than stop index",
    ):
        np.testing.assert_array_almost_equal(
            Polyline(v=example_vs, is_closed=False).sliced_at_indices(3, 3).v,
            np.array(
                [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
            ),
        )


def test_polyline_nearest():
    def as_3d(points_2d):
        return np.hstack([points_2d, np.repeat(-2.5, len(points_2d)).reshape(-1, 1)])

    chomper = Polyline(
        v=as_3d(
            np.array(
                [
                    [2, 4],
                    [4, 3],
                    [7, 2],
                    [6, 6],
                    [9, 5],
                    [10, 9],
                    [7, 7],
                    [7, 8],
                    [10, 9],
                    [7, 10],
                    [4, 10],
                    [1, 8],
                    [3, 8],
                    [2, 9],
                    [1, 8],
                ]
            )
        ),
        is_closed=True,
    )

    query_points = as_3d(np.array([[2.5, 7.5], [6, -7], [7, 3], [17, 8]]))
    expected_segment_indices = np.array([11, 1, 2, 4])
    # [115./17., 14./17.] seems right, could probably be verified using the
    # formulas at https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    expected_closest_points = as_3d(
        np.array([[2.5, 8], [7, 2], [115.0 / 17.0, 50.0 / 17.0], [10, 9]])
    )

    points, segment_indices = chomper.nearest(query_points, ret_segment_indices=True)
    np.testing.assert_array_equal(segment_indices, expected_segment_indices)
    np.testing.assert_array_almost_equal(points, expected_closest_points)
    np.testing.assert_array_almost_equal(
        chomper.nearest(query_points, ret_segment_indices=False),
        expected_closest_points,
    )

    point, segment_index = chomper.nearest(query_points[0], ret_segment_indices=True)
    assert segment_index == expected_segment_indices[0]
    np.testing.assert_array_almost_equal(point, expected_closest_points[0])
    np.testing.assert_array_almost_equal(
        chomper.nearest(query_points[0], ret_segment_indices=False),
        expected_closest_points[0],
    )


def test_slice_at_points():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    start_point = np.array([0.5, 0.0, 0.0])
    end_point = np.array([1.0, 1.5, 0.0])
    sliced = Polyline(v=points, is_closed=False).sliced_at_points(
        start_point, end_point
    )
    assert sliced.is_closed is False
    np.testing.assert_array_equal(
        sliced.v, np.array([start_point, [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], end_point]),
    )

    sliced = Polyline(v=points, is_closed=True).sliced_at_points(end_point, start_point)
    assert sliced.is_closed is False
    np.testing.assert_array_equal(
        sliced.v, np.array([end_point, [1.0, 2.0, 0.0], [0.0, 0.0, 0.0], start_point]),
    )

    sliced = Polyline(v=points, is_closed=True).sliced_at_points(
        end_point, np.array([0.0, 0.0, 0.0])
    )
    assert sliced.is_closed is False
    np.testing.assert_array_equal(
        sliced.v, np.array([end_point, [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    )

    sliced = Polyline(v=points).sliced_at_points(np.array([1.0, 1.0, 0.0]), end_point)
    assert sliced.is_closed is False
    np.testing.assert_array_equal(sliced.v, np.array([[1.0, 1.0, 0.0], end_point]))
