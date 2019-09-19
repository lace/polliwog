import numpy as np
import vg
import pytest
from .composite import CompositeTransform, convert_44_to_33


def create_cube_verts(origin, size):
    # Create a cube. Since CompositeTransform just works on verticies,
    # we don't need a full lace.mesh object.
    origin = np.asarray(origin)
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


def test_translate():
    transform = CompositeTransform()
    transform.translate(np.array([8.0, 6.0, 7.0]))

    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    transformed_cube_v = transform(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [9.0, 6.0, 7.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [13.0, 10.0, 11.0])


def test_translate_by_list():
    transform = CompositeTransform()
    transform.translate([8.0, 6.0, 7.0])

    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    transformed_cube_v = transform(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [9.0, 6.0, 7.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [13.0, 10.0, 11.0])


def test_scale():
    transform = CompositeTransform()
    transform.scale(10.0)

    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    transformed_cube_v = transform(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [10.0, 0.0, 0.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [50.0, 40.0, 40.0])


def test_scale_error():
    transform = CompositeTransform()
    with pytest.raises(ValueError):
        transform.scale(-1)


def test_convert_units():
    transform = CompositeTransform()
    transform.convert_units("m", "cm")

    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    transformed_cube_v = transform(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [100.0, 0.0, 0.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [500.0, 400.0, 400.0])


def test_translate_then_scale():
    transform = CompositeTransform()
    transform.translate(np.array([8.0, 6.0, 7.0]))
    transform.scale(10.0)

    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    transformed_cube_v = transform(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [90.0, 60.0, 70.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [130.0, 100.0, 110.0])


def test_scale_then_translate():
    transform = CompositeTransform()
    transform.scale(10.0)
    transform.translate(np.array([8.0, 6.0, 7.0]))

    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    transformed_cube_v = transform(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [18.0, 6.0, 7.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [58.0, 46.0, 47.0])


def test_rotate_then_translate():
    transform = CompositeTransform()
    transform.rotate(np.array([1.0, 2.0, 3.0]))
    transform.translate(np.array([3.0, 2.0, 1.0]))

    v = np.array([1.0, 0.0, 0.0]).reshape(-1, 3)

    # Forward.
    np.testing.assert_allclose(
        np.array([2.30507944, 1.80799303, 1.69297817]).reshape(-1, 3), transform(v)
    )
    # Reverse.
    np.testing.assert_allclose(
        np.array([1.08087689, -1.45082159, -2.3930779]).reshape(-1, 3),
        transform(v, reverse=True),
    )


def test_reorient():
    # TODO We should also test a non-axis-aligned up and look.

    transform = CompositeTransform()
    transform.reorient(up=vg.basis.y, look=vg.basis.neg_x)

    cube_v = create_default_cube_verts()

    # Confidence check.
    np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

    transformed_cube_v = transform(cube_v)

    np.testing.assert_array_equal(transformed_cube_v[0], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(transformed_cube_v[6], [4, 4.0, -5.0])


def test_rotate():
    ways_to_rotate_around_y_a_quarter_turn = [
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        np.array([0, np.pi / 2, 0]),
        np.array([[0, np.pi / 2, 0]]),
        np.array([[0], [np.pi / 2], [0]]),
        [0, np.pi / 2, 0],
    ]
    for rot in ways_to_rotate_around_y_a_quarter_turn:
        transform = CompositeTransform()
        transform.rotate(rot)
        cube_v = create_default_cube_verts()

        # Confidence check.
        np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

        transformed_cube_v = transform(cube_v)

        np.testing.assert_array_almost_equal(transformed_cube_v[0], [0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(transformed_cube_v[6], [4, 4.0, -5.0])


def test_reverse_transforms():
    transforms = [CompositeTransform() for _ in range(5)]

    transforms[1].translate(np.array([8.0, 6.0, 7.0]))

    transforms[2].scale(10.0)

    transforms[3].translate(np.array([8.0, 6.0, 7.0]))
    transforms[3].scale(10.0)

    transforms[4].scale(10.0)
    transforms[4].translate(np.array([8.0, 6.0, 7.0]))

    for transform in transforms:
        cube_v = create_default_cube_verts()

        # Confidence check.
        np.testing.assert_array_equal(cube_v[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(cube_v[6], [5.0, 4.0, 4.0])

        transformed = transform(cube_v)

        untransformed_v = transform(transformed, reverse=True)

        np.testing.assert_array_almost_equal(untransformed_v[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(untransformed_v[6], [5.0, 4.0, 4.0])


def test_forward_reverse_equivalence():
    transform = CompositeTransform()
    transform.rotate(np.array([1.0, 2.0, 3.0]))
    transform.translate(np.array([3.0, 2.0, 1.0]))
    transform.scale(10.0)
    transform.rotate(np.array([7.0, 13.0, 5.0]))

    forward = transform.matrix_for()
    reverse = transform.matrix_for(reverse=True)
    np.testing.assert_allclose(reverse, np.linalg.inv(forward))

    forward = transform.matrix_for(from_range=(0, 2))
    reverse = transform.matrix_for(from_range=(0, 2), reverse=True)
    np.testing.assert_allclose(reverse, np.linalg.inv(forward))


def test_convert_44_to_33():
    np.testing.assert_array_equal(
        convert_44_to_33(
            np.array(
                [
                    [1.0, 2.0, 3.0, 0.0],
                    [2.0, 3.0, 4.0, 0.0],
                    [5.0, 6.0, 7.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        ),
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
    )
