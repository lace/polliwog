import numpy as np
from vg.compat import v2 as vg
from ._affine_transform import (
    transform_matrix_for_rotation,
    transform_matrix_for_translation,
)
from ._apply import compose_transforms


def world_to_view(position, target, up=vg.basis.y, inverse=False):
    """
    Create a transform matrix which sends world-space coordinates to
    view-space coordinates.

    Args:
        position (np.ndarray): The camera's position in world coordinates.
        target (np.ndarray): The camera's target in world coordinates.
            `target - position` is the "look at" vector.
        up (np.ndarray): The approximate up direction, in world coordinates.
        inverse (bool): When `True`, return the inverse transform instead.

    Returns:
        np.ndarray: The `4x4` transformation matrix, which can be used with
        `polliwog.transform.apply_transform()`.

    See also:
        https://cseweb.ucsd.edu/classes/wi18/cse167-a/lec4.pdf
        http://www.songho.ca/opengl/gl_camera.html
    """
    vg.shape.check(locals(), "position", (3,))
    vg.shape.check(locals(), "target", (3,))

    look = vg.normalize(target - position)
    left = vg.normalize(vg.cross(look, up))
    recomputed_up = vg.cross(left, look)

    rotation = transform_matrix_for_rotation(np.array([left, recomputed_up, look]))
    if inverse:
        inverse_rotation = rotation.T
        inverse_translation = transform_matrix_for_translation(position)
        return compose_transforms(inverse_rotation, inverse_translation)
    else:
        translation = transform_matrix_for_translation(-position)
        return compose_transforms(translation, rotation)


def view_to_orthographic_projection(width, height, near=0.1, far=2000, inverse=False):
    """
    Create an orthographic projection matrix with the given parameters, which
    maps points from world space to coordinates in the normalized view volume.
    These coordinates range from -1 to 1 in x, y, and z with `(-1, -1, -1)`
    at the bottom-left of the near clipping plane, and `(1, 1, 1)` at the
    top-right of the far clipping plane.

    Args:
        width (float): Width of the window, in pixels. (FIXME: Is this really
            correct?)
        height (float): Height of the window, in pixels. (FIXME: Is this really
            correct?)
        near (float): Near clipping plane. (FIXME: Clarify!)
        far (float): Far clipping plane. (FIXME: Clarify!)
        inverse (bool): When `True`, return the inverse transform instead.

    Returns:
        np.ndarray: The `4x4` transformation matrix, which can be used with
        `polliwog.transform.apply_transform()`.

    See also:
        https://cseweb.ucsd.edu/classes/wi18/cse167-a/lec4.pdf
        http://www.songho.ca/opengl/gl_projectionmatrix.html
        http://glasnost.itcarlow.ie/~powerk/GeneralGraphicsNotes/projection/orthographicprojection.html
    """
    if inverse:
        inverse_translate = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, (far + near) / (far - near)],
                [0, 0, 0, 1],
            ]
        )
        inverse_scale = np.array(
            [
                [width / 2, 0, 0, 0],
                [0, height / 2, 0, 0],
                [0, 0, (far - near) / -2, 0],
                [0, 0, 0, 1],
            ]
        )
        return compose_transforms(inverse_translate, inverse_scale)
    else:
        scale = np.array(
            [
                [2 / width, 0, 0, 0],
                [0, 2 / height, 0, 0],
                [0, 0, -2 / (far - near), 0],
                [0, 0, 0, 1],
            ]
        )
        translate = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -(far + near) / (far - near)],
                [0, 0, 0, 1],
            ]
        )
        return compose_transforms(scale, translate)


def viewport_transform(x_right, y_bottom, x_left=0, y_top=0, inverse=False):
    """
    Create a matrix which transforms from the normalized view volume to
    screen coordinates, with a depth value ranging from 0 in front to 1 in
    back.

    No clipping is performed.

    Args:
        x_right (int): The `x` coordinate of the right of the viewport.
            (usually the width).
        y_bottom (int): The `y` coordinate of the bottom of the viewport
            (usually the height).
        x_left (int): The `x` coordinate of the left of the viewport
            (usually zero).
        y_top (int): The `y` coordinate of the top of the viewport
            (usually zero).
        inverse (bool): When `True`, return the inverse transform instead.

    Returns:
        np.ndarray: The `4x4` transformation matrix, which can be used with
        `polliwog.transform.apply_transform()`.

    See also:
        https://cseweb.ucsd.edu/classes/wi18/cse167-a/lec4.pdf
        http://glasnost.itcarlow.ie/~powerk/GeneralGraphicsNotes/projection/viewport_transformation.html
    """
    if inverse:
        inverse_translate = np.array(
            [
                [1, 0, 0, -0.5 * (x_right + x_left)],
                [0, 1, 0, -0.5 * (y_top + y_bottom)],
                [0, 0, 1, -0.5],
                [0, 0, 0, 1],
            ]
        )
        inverse_scale = np.array(
            [
                [2 / (x_right - x_left), 0, 0, 0],
                [0, 2 / (y_top - y_bottom), 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 1],
            ]
        )
        return compose_transforms(inverse_translate, inverse_scale)
    else:
        scale = np.array(
            [
                [0.5 * (x_right - x_left), 0, 0, 0],
                [0, 0.5 * (y_top - y_bottom), 0, 0],
                [0, 0, 0.5, 0],
                [0, 0, 0, 1],
            ]
        )
        translate = np.array(
            [
                [1, 0, 0, 0.5 * (x_right + x_left)],
                [0, 1, 0, 0.5 * (y_top + y_bottom)],
                [0, 0, 1, 0.5],
                [0, 0, 0, 1],
            ]
        )
        return compose_transforms(scale, translate)


def world_to_canvas_orthographic_projection(
    width, height, position, target, zoom=1, inverse=False
):
    """
    Create a transformation matrix which composes camera, orthographic
    projection, and viewport transformations into a single operation.

    Args:
        width (float): Width of the window, in pixels. (FIXME: Is this really
            correct?)
        height (float): Height of the window, in pixels. (FIXME: Is this really
            correct?)
        position (np.ndarray): The camera's position in world coordinates.
        target (np.ndarray): The camera's target in world coordinates.
            `target - position` is the "look at" vector.
        inverse (bool): When `True`, return the inverse transform instead.

    Returns:
        np.ndarray: The `4x4` transformation matrix, which can be used with
        `polliwog.transform.apply_transform()`.

    """
    from ._apply import compose_transforms

    transforms = [
        world_to_view(position=position, target=target, inverse=inverse),
        view_to_orthographic_projection(
            width=width / zoom, height=height / zoom, inverse=inverse
        ),
        viewport_transform(x_right=width, y_bottom=height, inverse=inverse),
    ]
    if inverse:
        transforms.reverse()
    return compose_transforms(*transforms)
