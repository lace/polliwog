import numpy as np
import vg
import pytest
from ..plane.plane import Plane
from .polyline import Polyline


def test_repr():
    example_vs = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    assert repr(Polyline(example_vs, closed=True)) == "<closed Polyline with 4 verts>"
    assert repr(Polyline(example_vs, closed=False)) == "<open Polyline with 4 verts>"
    assert (
        repr(Polyline(np.array([]).reshape(-1, 3), closed=False))
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
    actual_dict = Polyline(example_vs, closed=True).to_dict(decimals=3)
    # TODO Is there a cleaner way to assert deep equal?
    assert set(actual_dict.keys()) == set(expected_dict.keys())
    np.testing.assert_array_equal(expected_dict["vertices"], actual_dict["vertices"])
    np.testing.assert_array_equal(expected_dict["edges"], actual_dict["edges"])


def test_update_closed():
    example_vs = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    polyline = Polyline(example_vs, closed=False)
    assert polyline.num_e == 3
    assert polyline.closed == False
    polyline.closed = True
    assert polyline.num_e == 4
    assert polyline.closed == True


def test_num_v_num_e():
    example_vs = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    closed_polyline = Polyline(example_vs, closed=True)
    assert closed_polyline.num_v == 4
    assert closed_polyline.num_e == 4
    open_polyline = Polyline(example_vs, closed=False)
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

    np.testing.assert_array_equal(Polyline(v, closed=True).e, expected_closed)


def test_segments():
    polyline = Polyline(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]),
        closed=True,
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
        closed=True,
    )

    expected = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -2.0, 0.0]]
    )

    np.testing.assert_array_equal(polyline.segment_vectors, expected)


def test_length_of_empty_polyline():
    polyline = Polyline(None)
    assert polyline.total_length == 0

    polyline = Polyline(None, closed=True)
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
        closed=True,
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
        closed=True,
    )

    expected_indices = np.array([0, 1, 2, 5, 6, 7])

    for max_length in (2.0, 2.5, 2.6):
        result = original.copy()
        indices = result.partition_by_length(max_length, ret_indices=True)

        np.testing.assert_array_almost_equal(result.v, expected.v)
        np.testing.assert_array_equal(result.e, expected.e)
        np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_bisect_edges():
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

    result = original.copy()
    indices = result.bisect_edges([1, 2])

    np.testing.assert_array_almost_equal(result.v, expected.v)
    np.testing.assert_array_equal(result.e, expected.e)
    np.testing.assert_array_equal(indices, expected_indices_of_original_vertices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_flip():
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
        closed=True,
    )

    expected = Polyline(
        np.array(
            [
                [0.0, 8.0, 0.0],
                [1.0, 8.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        closed=True,
    )

    original.flip()

    np.testing.assert_array_almost_equal(original.v, expected.v)


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
        closed=True,
    )

    reindexed, edge_mapping = original.reindexed(5, ret_edge_mapping=True)

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
        closed=True,
    )

    np.testing.assert_array_almost_equal(reindexed.v, expected.v)
    np.testing.assert_array_equal(original.segments[edge_mapping], reindexed.segments)

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
        closed=False,
    )
    with pytest.raises(ValueError):
        open_polyline.reindexed(5)


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
        closed=True,
    )

    expected = np.array([[1.0, 7.5, 0.0], [0.0, 7.5, 0.0]])
    actual = polyline.intersect_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.y)
    )

    np.testing.assert_array_equal(actual, expected)


def test_cut_by_plane_closed():
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
        closed=True,
    )

    expected = Polyline(
        np.array([[1.0, 7.5, 0.0], [1.0, 8.0, 0.0], [0.0, 8.0, 0.0], [0.0, 7.5, 0.0]]),
        closed=False,
    )
    actual = original.cut_by_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.y)
    )

    np.testing.assert_array_almost_equal(actual.v, expected.v)
    assert actual.closed == False

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
        closed=False,
    )
    actual = original.cut_by_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.neg_y)
    )

    np.testing.assert_array_almost_equal(actual.v, expected.v)
    assert actual.closed == False

    zigzag = Polyline(
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [5.0, 2.0, 0.0]]),
        closed=True,
    )
    with pytest.raises(ValueError):
        zigzag.cut_by_plane(
            Plane(point_on_plane=np.array([2.5, 0.0, 0.0]), unit_normal=vg.basis.x)
        )


# def test_cut_by_polyline_closed_on_vertex():
#     # TODO: This isn't working right now.
#     original = Polyline(
#         np.array(
#             [
#                 [0.0, 0.0, 0.0],
#                 [1.0, 0.0, 0.0],
#                 [1.0, 1.0, 0.0],
#                 [1.0, 7.0, 0.0],
#                 [1.0, 8.0, 0.0],
#                 [0.0, 8.0, 0.0],
#             ]
#         ),
#         closed=True,
#     )
#     expected = Polyline(
#         np.array(
#             [
#                 [0.0, 7.5, 0.0],
#                 [0.0, 0.0, 0.0],
#                 [1.0, 0.0, 0.0],
#                 [1.0, 1.0, 0.0],
#                 [1.0, 7.0, 0.0],
#                 [1.0, 7.5, 0.0],
#             ]
#         ),
#         closed=False,
#     )
#     actual = original.cut_by_plane(
#         Plane(point_on_plane=np.array([0.0, 1.0, 0.0]), unit_normal=vg.basis.y)
#     )
#     np.testing.assert_array_almost_equal(actual.v, expected.v)
#     assert actual.closed == False


def test_cut_by_plane_open():
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
        closed=False,
    )

    expected = Polyline(np.array([[1.0, 7.5, 0.0], [1.0, 8.0, 0.0]]), closed=False)
    actual = original.cut_by_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.y)
    )

    np.testing.assert_array_almost_equal(actual.v, expected.v)
    assert actual.closed == False

    expected = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
                [1.0, 7.5, 0.0],
            ]
        ),
        closed=False,
    )
    actual = original.cut_by_plane(
        Plane(point_on_plane=np.array([0.0, 7.5, 0.0]), unit_normal=vg.basis.neg_y)
    )

    np.testing.assert_array_almost_equal(actual.v, expected.v)
    assert actual.closed == False

    with pytest.raises(ValueError):
        original.cut_by_plane(
            Plane(point_on_plane=np.array([0.0, 15.0, 0.0]), unit_normal=vg.basis.neg_y)
        )

    with pytest.raises(ValueError):
        original.cut_by_plane(
            Plane(
                point_on_plane=np.array([0.5, 0.0, 0.0]),
                unit_normal=vg.normalize(np.array([1.0, -1.0, 0.0])),
            )
        )


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
        Polyline(v, closed=False).apex(vg.basis.y), np.array([1.0, 3.0, 0.0])
    )
