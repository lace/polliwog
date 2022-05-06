# Imported and adapated from Trimesh
#
# Copyright (c) 2019 Michael Dawson-Haggerty
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# https://github.com/mikedh/trimesh/blob/510b4468d65ecb52759c9c660bf5c4791361d63f/tests/test_section.py#L156-L234

import numpy as np
from polliwog.plane import slice_triangles_by_plane
from vg.compat import v2 as vg
from ..shapes import rectangular_prism
from ..tri import FACE_DTYPE, surface_normals

cube_origin = np.array([-0.5, -0.5, -0.5])
cube_vertices, cube_faces = rectangular_prism(
    origin=cube_origin,
    size=np.repeat(1, 3),
    ret_unique_vertices_and_faces=True,
)
# Note: Orientation of `cube_faces`:
# 0, 1: -y
# 2, 3: +y
# 4, 5: -z
# 6, 7: +x
# 8, 9: +z
# 10, 11: -x
cube_extent = np.max(cube_vertices, axis=0)


def test_slice_cube_corner():
    common_kwargs = dict(
        vertices=cube_vertices,
        faces=cube_faces,
        plane_reference_point=cube_extent - 0.05,
        plane_normal=np.array([1, 1, 1]),
    )

    sliced_vertices, sliced_faces = slice_triangles_by_plane(
        **common_kwargs,
    )

    np.testing.assert_array_almost_equal(
        np.min(sliced_vertices, axis=0), cube_extent - 0.15
    )
    np.testing.assert_array_almost_equal(np.max(sliced_vertices, axis=0), cube_extent)
    assert len(sliced_faces) == 4

    sliced_vertices, sliced_faces, face_mapping = slice_triangles_by_plane(
        **common_kwargs,
        ret_face_mapping=True,
    )

    # Check that the `face_mapping` is correct by seeing that the new faces
    # correspond to the expected `cube_faces`. See mapping above.

    # Confidence check.
    np.testing.assert_array_equal(
        surface_normals(sliced_vertices[sliced_faces]),
        np.array([vg.basis.y, vg.basis.x, vg.basis.z, vg.basis.z]),
    )

    # Assert.
    np.testing.assert_array_equal(
        face_mapping,
        np.array([2, 6, 8, 9]),
    )


def test_slice_cube_submesh():
    # Only act on the top of the cube.
    mask = np.zeros(len(cube_faces), dtype=np.bool)
    mask[[2, 3]] = True

    _, sliced_faces = slice_triangles_by_plane(
        vertices=cube_vertices,
        faces=cube_faces,
        plane_reference_point=cube_extent - 0.05,
        plane_normal=np.array([1, 1, 1]),
        faces_to_slice=mask,
    )
    # Only the corner of the top is kept, along with the rest of the cube.
    # TODO: Improve this assertion. For now, verify visually.
    assert len(sliced_faces) == 11


def test_slice_cube_top():
    """
    Tests new quads and entirely contained triangles.
    """
    common_kwargs = dict(
        vertices=cube_vertices,
        faces=cube_faces,
        plane_reference_point=cube_extent - 0.05,
        plane_normal=vg.basis.z,
    )

    sliced_vertices, sliced_faces = slice_triangles_by_plane(**common_kwargs)

    np.testing.assert_array_almost_equal(
        np.min(sliced_vertices, axis=0), cube_origin + np.array([0, 0, 0.95])
    )
    np.testing.assert_array_almost_equal(np.max(sliced_vertices, axis=0), cube_extent)
    assert len(sliced_faces) == 14

    sliced_vertices, sliced_faces, face_mapping = slice_triangles_by_plane(
        **common_kwargs, ret_face_mapping=True
    )

    # Check that the `face_mapping` is correct by seeing that the new faces
    # correspond to the expected `cube_faces`. See mapping above.

    # Confidence check.
    np.testing.assert_array_equal(
        surface_normals(sliced_vertices[sliced_faces]),
        np.array(
            [
                vg.basis.z,
                vg.basis.z,
                vg.basis.neg_y,
                vg.basis.neg_y,
                vg.basis.y,
                vg.basis.y,
                vg.basis.x,
                vg.basis.x,
                vg.basis.neg_x,
                vg.basis.neg_x,
                vg.basis.neg_y,
                vg.basis.y,
                vg.basis.x,
                vg.basis.neg_x,
            ]
        ),
    )

    # Assert.
    np.testing.assert_array_equal(
        face_mapping, np.array([8, 9, 1, 1, 2, 2, 6, 6, 10, 10, 0, 3, 7, 11])
    )


def test_slice_cube_edge_multiple_planes():
    sliced_vertices, sliced_faces = slice_triangles_by_plane(
        *slice_triangles_by_plane(
            vertices=cube_vertices,
            faces=cube_faces,
            plane_reference_point=cube_extent - 0.05,
            plane_normal=vg.basis.z,
        ),
        plane_reference_point=cube_extent - 0.05,
        plane_normal=vg.basis.y,
    )

    np.testing.assert_array_almost_equal(
        np.min(sliced_vertices, axis=0), cube_origin + np.array([0, 0.95, 0.95])
    )
    np.testing.assert_array_almost_equal(np.max(sliced_vertices, axis=0), cube_extent)
    assert len(sliced_faces) == 12


def test_slice_cube_all_in_front():
    common_kwargs = dict(
        vertices=cube_vertices,
        faces=cube_faces,
        plane_reference_point=cube_origin,
        plane_normal=vg.basis.x,
    )

    sliced_vertices, sliced_faces = slice_triangles_by_plane(**common_kwargs)

    np.testing.assert_array_equal(sliced_vertices, cube_vertices)
    np.testing.assert_array_equal(sliced_faces, cube_faces)

    sliced_vertices, sliced_faces, face_mapping = slice_triangles_by_plane(
        **common_kwargs,
        ret_face_mapping=True,
    )

    np.testing.assert_array_equal(sliced_vertices, cube_vertices)
    np.testing.assert_array_equal(sliced_faces, cube_faces)
    np.testing.assert_array_equal(face_mapping, np.arange(len(cube_faces)))


def test_slice_cube_all_in_back():
    common_kwargs = dict(
        vertices=cube_vertices,
        faces=cube_faces,
        plane_reference_point=cube_origin + 5,
        plane_normal=vg.basis.x,
    )

    sliced_vertices, sliced_faces = slice_triangles_by_plane(**common_kwargs)

    assert len(sliced_vertices) == 0
    assert len(sliced_faces) == 0

    sliced_vertices, sliced_faces, face_mapping = slice_triangles_by_plane(
        **common_kwargs, ret_face_mapping=True
    )

    assert len(sliced_vertices) == 0
    assert len(sliced_faces) == 0
    assert len(face_mapping) == 0


def test_slice_empty():
    common_kwargs = dict(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=FACE_DTYPE),
        plane_reference_point=np.zeros(3),
        plane_normal=vg.basis.x,
    )

    sliced_vertices, sliced_faces = slice_triangles_by_plane(**common_kwargs)

    assert len(sliced_vertices) == 0
    assert len(sliced_faces) == 0

    sliced_vertices, sliced_faces, face_mapping = slice_triangles_by_plane(
        **common_kwargs,
        ret_face_mapping=True,
    )

    assert len(sliced_vertices) == 0
    assert len(sliced_faces) == 0
    assert len(face_mapping) == 0
