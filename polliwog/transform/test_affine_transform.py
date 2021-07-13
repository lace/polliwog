import numpy as np
from polliwog.transform import (
    apply_transform,
    transform_matrix_for_non_uniform_scale,
    transform_matrix_for_rotation,
    transform_matrix_for_translation,
    transform_matrix_for_uniform_scale,
)
import pytest


def create_cube_verts(origin, size):
    # Create a cube. Since CompositeTransform just works on verticies,
    # we don't need a full lace.mesh object.
    origin = np.array(origin)
    size = np.repeat(size, 3)
    lower_base_plane = np.array(
        [
            # Lower base plane
            origin,
            origin + np.array([size[0], 0, 0]),
            origin + np.array([size[0], 0, size[2]]),
            origin + np.array([0, 0, size[2]]),
        ]
    )
    upper_base_plane = lower_base_plane + np.array([0, size[1], 0])
    return np.vstack([lower_base_plane, upper_base_plane])


def create_default_cube_verts():
    return create_cube_verts([1.0, 0.0, 0.0], 4.0)


def test_rotate():
    cube_v = create_default_cube_verts()
    ways_to_rotate_around_y_a_quarter_turn = [
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        np.array([0, np.pi / 2, 0]),
    ]
    for rot in ways_to_rotate_around_y_a_quarter_turn:
        # Confidence check.
        np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

        transformed_cube_v = apply_transform(transform_matrix_for_rotation(rot))(cube_v)

        np.testing.assert_array_almost_equal(transformed_cube_v[0], [0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(transformed_cube_v[6], [4, 4.0, -5.0])


def test_translate():
    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    transformed_cube_v = apply_transform(
        transform_matrix_for_translation(np.array([8.0, 6.0, 7.0]))
    )(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [9.0, 6.0, 7.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [13.0, 10.0, 11.0])


def test_uniform_scale():
    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    transformed_cube_v = apply_transform(
        transform_matrix_for_uniform_scale(-10.0, allow_flipping=True)
    )(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [-10.0, 0.0, 0.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [-50.0, -40.0, -40.0])


def test_uniform_scale_error():
    with pytest.raises(ValueError, match="Scale factor should be nonzero"):
        transform_matrix_for_uniform_scale(0)
    with pytest.raises(ValueError, match="Scale factor should be greater than zero"):
        transform_matrix_for_uniform_scale(-1)


def test_non_uniform_scale():
    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    forward, inverse = transform_matrix_for_non_uniform_scale(
        1.0, -2.0, 1.0, allow_flipping=True, ret_inverse_matrix=True
    )
    transformed_cube_v = apply_transform(forward)(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [5.0, -8.0, 4.0])

    untransformed_cube_v = apply_transform(inverse)(transformed_cube_v)

    np.testing.assert_array_almost_equal(untransformed_cube_v, cube_v)

    np.testing.assert_array_equal(
        transform_matrix_for_non_uniform_scale(
            1.0, -2.0, 1.0, allow_flipping=True, ret_inverse_matrix=False
        ),
        forward,
    )


def test_non_uniform_scale_error():
    with pytest.raises(ValueError, match="Scale factors should be nonzero"):
        transform_matrix_for_non_uniform_scale(1.0, 0.0, 3.0)
    with pytest.raises(ValueError, match="Scale factors should be greater than zero"):
        transform_matrix_for_non_uniform_scale(1.0, -1.0, 3.0)
