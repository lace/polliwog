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


def test_slice_cube_corner():
    origin = np.array([-0.5, -0.5, -0.5])
    vertices, faces = rectangular_prism(
        origin=origin,
        size=np.repeat(1, 3),
        ret_unique_vertices_and_faces=True,
    )
    extent = np.max(vertices, axis=0)

    sliced_vertices, sliced_faces = slice_triangles_by_plane(
        vertices=vertices,
        faces=faces,
        point_on_plane=extent - 0.05,
        plane_normal=np.array([1, 1, 1]),
    )

    np.testing.assert_array_almost_equal(np.min(sliced_vertices, axis=0), extent - 0.15)
    np.testing.assert_array_almost_equal(np.max(sliced_vertices, axis=0), extent)
    assert len(sliced_faces) == 4


def test_slice_cube_submesh():
    origin = np.array([-0.5, -0.5, -0.5])
    vertices, faces = rectangular_prism(
        origin=origin,
        size=np.repeat(1, 3),
        ret_unique_vertices_and_faces=True,
    )
    extent = np.max(vertices, axis=0)

    # Only act on the top of the cube.
    mask = np.zeros(len(faces), dtype=np.bool)
    mask[[2, 3]] = True

    _, sliced_faces = slice_triangles_by_plane(
        vertices=vertices,
        faces=faces,
        point_on_plane=extent - 0.05,
        plane_normal=np.array([1, 1, 1]),
        faces_to_slice=mask,
    )
    # Only the corner of the top is kept, along with the rest of the cube.
    # TODO: Improve this assertion.
    assert len(sliced_faces) == 11


def test_slice_cube_top():
    """
    Tests new quads and entirely contained triangles.
    """
    origin = np.array([-0.5, -0.5, -0.5])
    vertices, faces = rectangular_prism(
        origin=origin,
        size=np.repeat(1, 3),
        ret_unique_vertices_and_faces=True,
    )
    extent = np.max(vertices, axis=0)

    sliced_vertices, sliced_faces = slice_triangles_by_plane(
        vertices=vertices,
        faces=faces,
        point_on_plane=extent - 0.05,
        plane_normal=vg.basis.z,
    )

    np.testing.assert_array_almost_equal(
        np.min(sliced_vertices, axis=0), origin + np.array([0, 0, 0.95])
    )
    np.testing.assert_array_almost_equal(np.max(sliced_vertices, axis=0), extent)
    assert len(sliced_faces) == 14


def test_slice_cube_edge_multiple_planes():
    origin = np.array([-0.5, -0.5, -0.5])
    vertices, faces = rectangular_prism(
        origin=origin,
        size=np.repeat(1, 3),
        ret_unique_vertices_and_faces=True,
    )
    extent = np.max(vertices, axis=0)

    sliced_vertices, sliced_faces = slice_triangles_by_plane(
        *slice_triangles_by_plane(
            vertices=vertices,
            faces=faces,
            point_on_plane=extent - 0.05,
            plane_normal=vg.basis.z,
        ),
        point_on_plane=extent - 0.05,
        plane_normal=vg.basis.y,
    )

    np.testing.assert_array_almost_equal(
        np.min(sliced_vertices, axis=0), origin + np.array([0, 0.95, 0.95])
    )
    np.testing.assert_array_almost_equal(np.max(sliced_vertices, axis=0), extent)
    assert len(sliced_faces) == 12


def test_slice_cube_all_in_front():
    origin = np.array([-0.5, -0.5, -0.5])
    vertices, faces = rectangular_prism(
        origin=origin,
        size=np.repeat(1, 3),
        ret_unique_vertices_and_faces=True,
    )

    sliced_vertices, sliced_faces = slice_triangles_by_plane(
        vertices=vertices,
        faces=faces,
        point_on_plane=origin,
        plane_normal=vg.basis.x,
    )

    np.testing.assert_array_equal(sliced_vertices, vertices)
    np.testing.assert_array_equal(sliced_faces, faces)


def test_slice_cube_all_in_back():
    origin = np.array([-0.5, -0.5, -0.5])
    vertices, faces = rectangular_prism(
        origin=origin,
        size=np.repeat(1, 3),
        ret_unique_vertices_and_faces=True,
    )

    sliced_vertices, sliced_faces = slice_triangles_by_plane(
        vertices=vertices,
        faces=faces,
        point_on_plane=origin + 5,
        plane_normal=vg.basis.x,
    )

    assert len(sliced_vertices) == 0
    assert len(sliced_faces) == 0


def test_slice_empty():
    vertices = np.zeros((0, 3), dtype=np.float64)
    faces = np.zeros((0, 3), dtype=np.uint64)

    sliced_vertices, sliced_faces = slice_triangles_by_plane(
        vertices=vertices,
        faces=faces,
        point_on_plane=np.zeros(3),
        plane_normal=vg.basis.x,
    )

    np.testing.assert_array_equal(sliced_vertices, vertices)
    np.testing.assert_array_equal(sliced_faces, faces)
