import numpy as np
import pytest
import vg
from ._coordinate_manager import CoordinateManager
from .test_affine_transform import create_cube_verts


def perform_transform_test(apply_transform_fn, expected_v0, expected_v6):
    cube_v = create_cube_verts([1.0, 0.0, 0.0], 4.0)

    coordinate_manager = CoordinateManager()
    coordinate_manager.tag_as("before")
    apply_transform_fn(coordinate_manager)
    coordinate_manager.tag_as("after")

    # Confidence check.
    np.testing.assert_array_almost_equal(cube_v[0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_almost_equal(cube_v[6], np.array([5.0, 4.0, 4.0]))

    coordinate_manager.before = cube_v
    scaled_v = coordinate_manager.do_transform(
        cube_v, from_tag="before", to_tag="after"
    )

    np.testing.assert_array_almost_equal(scaled_v[0], expected_v0)
    np.testing.assert_array_almost_equal(scaled_v[6], expected_v6)


def test_coordinate_manager_forward():
    cube_v = create_cube_verts([1.0, 0.0, 0.0], 4.0)
    cube_floor_point = np.array([3.0, 0.0, 2.0])  # as lace.mesh.floor_point

    coordinate_manager = CoordinateManager()
    coordinate_manager.tag_as("source")
    coordinate_manager.translate(-cube_floor_point)
    coordinate_manager.scale(2)
    coordinate_manager.tag_as("floored_and_scaled")
    coordinate_manager.translate(np.array([0.0, -4.0, 0.0]))
    coordinate_manager.tag_as("centered_at_origin")

    coordinate_manager.source = cube_v

    floored_and_scaled_v = coordinate_manager.do_transform(
        cube_v, from_tag="source", to_tag="floored_and_scaled"
    )

    # Confidence check.
    np.testing.assert_array_almost_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(cube_v[6], [5.0, 4.0, 4.0])

    np.testing.assert_array_almost_equal(floored_and_scaled_v[0], [-4.0, 0.0, -4.0])
    np.testing.assert_array_almost_equal(floored_and_scaled_v[6], [4.0, 8.0, 4.0])

    centered_at_origin_v_1 = coordinate_manager.do_transform(
        cube_v, from_tag="source", to_tag="centered_at_origin"
    )
    centered_at_origin_v_2 = coordinate_manager.do_transform(
        floored_and_scaled_v, from_tag="floored_and_scaled", to_tag="centered_at_origin"
    )

    np.testing.assert_array_almost_equal(centered_at_origin_v_1[0], [-4.0, -4.0, -4.0])
    np.testing.assert_array_almost_equal(centered_at_origin_v_1[6], [4.0, 4.0, 4.0])

    np.testing.assert_array_almost_equal(centered_at_origin_v_2[0], [-4.0, -4.0, -4.0])
    np.testing.assert_array_almost_equal(centered_at_origin_v_2[6], [4.0, 4.0, 4.0])

    source_v_1 = coordinate_manager.do_transform(
        floored_and_scaled_v, from_tag="floored_and_scaled", to_tag="source"
    )
    source_v_2 = coordinate_manager.do_transform(
        centered_at_origin_v_1, from_tag="centered_at_origin", to_tag="source"
    )
    np.testing.assert_array_almost_equal(source_v_1, cube_v)
    np.testing.assert_array_almost_equal(source_v_2, cube_v)


def test_coordinate_manager_forward_with_attrs():
    cube_v = create_cube_verts([1.0, 0.0, 0.0], 4.0)
    cube_floor_point = np.array([3.0, 0.0, 2.0])  # as lace.mesh.floor_point

    coordinate_manager = CoordinateManager()
    coordinate_manager.tag_as("source")
    coordinate_manager.translate(-cube_floor_point)
    coordinate_manager.scale(2)
    coordinate_manager.tag_as("floored_and_scaled")
    coordinate_manager.translate(np.array([0.0, -4.0, 0.0]))
    coordinate_manager.tag_as("centered_at_origin")

    coordinate_manager.source = cube_v

    # Confidence check.
    np.testing.assert_array_almost_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(cube_v[6], [5.0, 4.0, 4.0])

    floored_and_scaled_v = coordinate_manager.floored_and_scaled
    np.testing.assert_array_almost_equal(floored_and_scaled_v[0], [-4.0, 0.0, -4.0])
    np.testing.assert_array_almost_equal(floored_and_scaled_v[6], [4.0, 8.0, 4.0])

    centered_at_origin_v = coordinate_manager.centered_at_origin
    np.testing.assert_array_almost_equal(centered_at_origin_v[0], [-4.0, -4.0, -4.0])
    np.testing.assert_array_almost_equal(centered_at_origin_v[6], [4.0, 4.0, 4.0])

    source_v = coordinate_manager.source
    np.testing.assert_array_almost_equal(source_v, cube_v)


def test_coordinate_manager_out_of_order():
    coordinate_manager = CoordinateManager()
    coordinate_manager.tag_as("before")
    coordinate_manager.scale(2)
    coordinate_manager.tag_as("after")

    with pytest.raises(ValueError):
        coordinate_manager.after


def test_coordinate_manager_invalid_tag():
    cube_v = create_cube_verts([1.0, 0.0, 0.0], 4.0)

    coordinate_manager = CoordinateManager()
    coordinate_manager.tag_as("before")
    coordinate_manager.scale(2)
    coordinate_manager.tag_as("after")

    with pytest.raises(AttributeError):
        coordinate_manager.beefour = cube_v

    coordinate_manager.before = cube_v

    with pytest.raises(KeyError):
        coordinate_manager.affturr


def test_coordinate_manager_custom_transform():
    scale = np.array([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])
    perform_transform_test(
        apply_transform_fn=lambda coordinate_manager: coordinate_manager.append_transform(
            scale
        ),
        expected_v0=np.array([3.0, 0.0, 0.0]),
        expected_v6=np.array([15.0, 12.0, 12.0]),
    )


def test_coordinate_manager_convert_units():
    perform_transform_test(
        apply_transform_fn=lambda coordinate_manager: coordinate_manager.convert_units(
            from_units="cm", to_units="m"
        ),
        expected_v0=np.array([0.01, 0.0, 0.0]),
        expected_v6=np.array([0.05, 0.04, 0.04]),
    )


def test_coordinate_manager_reorient():
    perform_transform_test(
        apply_transform_fn=lambda coordinate_manager: coordinate_manager.reorient(
            up=vg.basis.y, look=vg.basis.neg_x
        ),
        expected_v0=np.array([0.0, 0.0, -1.0]),
        expected_v6=np.array([4, 4.0, -5.0]),
    )


def test_coordinate_manager_rotate():
    perform_transform_test(
        apply_transform_fn=lambda coordinate_manager: coordinate_manager.rotate(
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        ),
        expected_v0=np.array([0.0, 0.0, -1.0]),
        expected_v6=np.array([4, 4.0, -5.0]),
    )
