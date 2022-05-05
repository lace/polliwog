import numpy as np
import pytest
from vg.compat import v2 as vg
from .. import Plane, Polyline


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


def test_serialize():
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
    ]

    serialized = Polyline(np.array(vertices), is_closed=True).serialize()

    assert serialized == {
        "vertices": vertices,
        "isClosed": True,
    }

    Polyline.validate(serialized)


def test_serialize_decimals():
    assert Polyline(
        np.array([[0.1234, 0.0, 0.0], [0.2345, 0.0, 0.0]]), is_closed=True
    ).serialize(decimals=1) == {
        "vertices": [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        "isClosed": True,
    }


def test_deserialize():
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
    ]

    deserialized = Polyline.deserialize(
        {
            "vertices": vertices,
            "isClosed": True,
        }
    )

    assert isinstance(deserialized, Polyline)
    np.testing.assert_array_equal(deserialized.v, np.array(vertices))
    assert deserialized.is_closed is True


def test_immutability():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    working_copy = vertices.copy()

    polyline = Polyline(v=working_copy, is_closed=False)
    np.testing.assert_array_equal(polyline.v, vertices)

    working_copy[0] = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_equal(polyline.v, vertices)


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
        points=points_to_insert,
        indices=before_indices,
    )
    assert polyline.is_closed is True
    np.testing.assert_array_almost_equal(polyline.v, expected_v)

    polyline, indices_of_original_vertices, indices_of_inserted_points = Polyline(
        v=original_points,
        is_closed=False,
    ).with_insertions(
        points=points_to_insert,
        indices=before_indices,
        ret_new_indices=True,
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
        v=original_points,
        is_closed=False,
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


def test_path_centroid():
    np.testing.assert_array_equal(
        Polyline(
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.0, 5.0, 0.0]]
            ),
            is_closed=True,
        ).path_centroid,
        np.array([0.5, 2.5, 0]),
    )


def test_length_of_empty_polyline():
    polyline = Polyline(np.zeros((0, 3)))
    assert polyline.total_length == 0

    polyline = Polyline(np.zeros((0, 3)), is_closed=True)
    assert polyline.total_length == 0


def test_subdivided_by_length_noop():
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

    result, indices = original.subdivided_by_length(1.0, ret_indices=True)

    expected_indices = np.array([0, 1, 2, 3, 4])

    np.testing.assert_array_almost_equal(result.v, original.v)
    np.testing.assert_array_equal(result.e, original.e)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_subdivided_by_length_degenerate():
    """
    This covers a bug that arose from a numerical stability issue in
    measurement on EC2 / MKL.
    """
    original = Polyline(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

    result, indices = original.subdivided_by_length(1.0, ret_indices=True)

    expected_indices = np.array([0, 1, 2])

    np.testing.assert_array_almost_equal(result.v, original.v)
    np.testing.assert_array_equal(result.e, original.e)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_subdivided_by_length_divide_by_two():
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
        result, indices = original.subdivided_by_length(max_length, ret_indices=True)

        np.testing.assert_array_almost_equal(result.v, expected.v)
        np.testing.assert_array_equal(result.e, expected.e)
        np.testing.assert_array_equal(indices, expected_indices)

        result_2 = result.subdivided_by_length(max_length, ret_indices=False)
        np.testing.assert_array_almost_equal(result_2.v, expected.v)
        assert result_2 is not result
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
        result, indices = original.subdivided_by_length(max_length, ret_indices=True)

        np.testing.assert_array_almost_equal(result.v, expected.v)
        np.testing.assert_array_equal(result.e, expected.e)
        np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_partition_length_divide_by_five_skip_some():
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

    edges_to_subdivide = np.array([False, True, False, True])

    expected = Polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.2, 0.0],
                [1.0, 0.4, 0.0],
                [1.0, 0.6, 0.0],
                [1.0, 0.8, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 2.2, 0.0],
                [1.0, 2.4, 0.0],
                [1.0, 2.6, 0.0],
                [1.0, 2.8, 0.0],
                [1.0, 3.0, 0.0],
            ]
        )
    )

    for max_length in (0.2, 0.24):
        result = original.subdivided_by_length(
            max_length, edges_to_subdivide=edges_to_subdivide
        )
        np.testing.assert_array_almost_equal(result.v, expected.v)
        np.testing.assert_array_equal(result.e, expected.e)


def test_subdivided_by_length_divide_some_leave_some():
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
        result, indices = original.subdivided_by_length(max_length, ret_indices=True)

        np.testing.assert_array_almost_equal(result.v, expected.v)
        np.testing.assert_array_equal(result.e, expected.e)
        np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(result.v[indices], original.v)


def test_subdivided_by_length_closed():
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
        result, indices = original.subdivided_by_length(max_length, ret_indices=True)

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

    flipped = original.flipped_if(True)

    assert flipped is not original
    np.testing.assert_array_almost_equal(flipped.v, expected_v)

    assert original.flipped_if(False) is original


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
        sliced.v,
        np.array([start_point, [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], end_point]),
    )

    sliced = Polyline(v=points, is_closed=True).sliced_at_points(end_point, start_point)
    assert sliced.is_closed is False
    np.testing.assert_array_equal(
        sliced.v,
        np.array([end_point, [1.0, 2.0, 0.0], [0.0, 0.0, 0.0], start_point]),
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


def test_sectioned():
    vs = np.arange(108).reshape(36, 3)
    polyline = Polyline(v=vs, is_closed=False)
    breakpoints = np.array([10, 15, 18, 27, 33])

    broken = polyline.sectioned(breakpoints)
    assert len(broken) == 6
    assert sum(section.total_length for section in broken) == polyline.total_length
    np.testing.assert_array_equal(broken[0].v, vs[:11])
    np.testing.assert_array_equal(broken[1].v, vs[10:16])
    np.testing.assert_array_equal(broken[2].v, vs[15:19])
    np.testing.assert_array_equal(broken[3].v, vs[18:28])
    np.testing.assert_array_equal(broken[4].v, vs[27:34])
    np.testing.assert_array_equal(broken[5].v, vs[33:])


def test_sectioned_degenerate():
    vs = np.arange(108).reshape(36, 3)
    polyline = Polyline(v=vs, is_closed=False)

    broken = polyline.sectioned(np.array([], dtype=np.int64))
    assert len(broken) == 1
    np.testing.assert_array_equal(broken[0].v, vs)


def test_sectioned_errors():
    vs = np.arange(108).reshape(36, 3)
    polyline = Polyline(v=vs, is_closed=False)
    with pytest.raises(ValueError, match="Every section must have at least one edge"):
        polyline.sectioned(np.array([0]))
    with pytest.raises(ValueError, match="Every section must have at least one edge"):
        polyline.sectioned(np.array([10, 10]))
    with pytest.raises(ValueError, match="Every section must have at least one edge"):
        polyline.sectioned(np.array([35]))

    with pytest.raises(
        NotImplementedError, match="Not implemented for closed polylines"
    ):
        Polyline(v=vs, is_closed=True).sectioned(np.array([10]))


def test_section_edge_case():
    vs = np.arange(108).reshape(36, 3)
    polyline = Polyline(v=vs, is_closed=False)
    broken = polyline.sectioned(np.array([10, 11]))
    assert len(broken) == 3
    assert sum(section.total_length for section in broken) == polyline.total_length
    np.testing.assert_array_equal(broken[0].v, vs[:11])
    np.testing.assert_array_equal(broken[1].v, vs[10:12])
    np.testing.assert_array_equal(broken[2].v, vs[11:])


def test_point_along_path():
    example_vs = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
            [2.0, 4.0, 0.0],
        ]
    )
    polyline = Polyline(v=example_vs, is_closed=False)
    fractions_of_total = np.array([0.2, 0.8, 0.05, 0.95, 1.0, 0.0])
    expected_points = np.array(
        [
            [1.0, 0.2, 0.0],
            [1.8, 3.0, 0.0],
            [0.3, 0.0, 0.0],
            [2.0, 3.7, 0.0],
            [2.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    for this_fraction_of_total, this_expected_point in zip(
        fractions_of_total, expected_points
    ):
        np.testing.assert_array_almost_equal(
            polyline.point_along_path(float(this_fraction_of_total)),
            this_expected_point,
        )

    np.testing.assert_array_almost_equal(
        polyline.point_along_path(fractions_of_total), expected_points
    )


def test_point_along_path_non_unit_length():
    start_point = np.array([-20, 125, -20])
    polyline = Polyline(
        v=np.array([start_point, [0, 125, 20], [20, 125, -20]]), is_closed=True
    )
    fraction_of_length = 0.009259259259259259
    point_along_path = polyline.point_along_path(fraction_of_length)
    np.testing.assert_array_almost_equal(
        polyline.nearest(point_along_path), point_along_path
    )
    np.testing.assert_almost_equal(
        vg.euclidean_distance(start_point, point_along_path),
        fraction_of_length * polyline.total_length,
    )


def test_point_along_path_errors():
    vs = np.arange(108).reshape(36, 3)
    polyline = Polyline(v=vs, is_closed=False)
    with pytest.raises(
        ValueError, match="fraction_of_total must be a value between 0 and 1"
    ):
        polyline.point_along_path(2.5)
