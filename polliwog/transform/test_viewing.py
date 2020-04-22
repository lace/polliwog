import numpy as np
from ._apply import apply_transform
from ._viewing import world_to_canvas_orthographic_projection


def teapot_verts():
    import os

    return np.load(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "teapot.npy")
    )


def test_teapot_orthographic_projection():
    world_coords = teapot_verts()
    canvas_coords = apply_transform(
        world_to_canvas_orthographic_projection(
            width=800,
            height=600,
            position=np.array([0.5, 1.5, 0.0]),
            target=np.array([-0.5, 1.0, -2.0]),
            zoom=100,
        )
    )(world_coords, discard_z_coord=True)

    expected_head = np.array(
        [[86.950483, 236.566495], [91.324232, 235.067513], [84.079372, 238.229429]]
    )
    expected_tail = np.array(
        [[661.899462, 233.196566], [661.899462, 233.196566], [662.424938, 233.68759]]
    )
    np.testing.assert_array_almost_equal(
        canvas_coords[0 : len(expected_head)], expected_head
    )
    np.testing.assert_array_almost_equal(
        canvas_coords[-len(expected_tail) :], expected_tail
    )


def test_inverse_orthographic_projection():
    world_coords = teapot_verts()
    canvas_coords = apply_transform(
        world_to_canvas_orthographic_projection(
            width=800,
            height=600,
            position=np.array([0.5, 1.5, 0.0]),
            target=np.array([-0.5, 1.0, -2.0]),
            zoom=100,
        )
    )(world_coords)
    untransformed_world_coords = apply_transform(
        world_to_canvas_orthographic_projection(
            width=800,
            height=600,
            position=np.array([0.5, 1.5, 0.0]),
            target=np.array([-0.5, 1.0, -2.0]),
            zoom=100,
            inverse=True,
        )
    )(canvas_coords)
    np.testing.assert_array_almost_equal(untransformed_world_coords, world_coords)


def generate_teapot_image():
    from polliwog.transform._testing_helper import write_canvas_points_to_png

    world_coords = teapot_verts()
    canvas_coords = apply_transform(
        world_to_canvas_orthographic_projection(
            width=800,
            height=600,
            position=np.array([0.5, 1.5, 0.0]),
            target=np.array([-0.5, 1.0, -2.0]),
            zoom=100,
        )
    )(world_coords, discard_z_coord=True)

    write_canvas_points_to_png(canvas_coords, 800, 600, "projected.png")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    generate_teapot_image()
