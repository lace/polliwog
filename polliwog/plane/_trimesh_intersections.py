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
# https://github.com/mikedh/trimesh/blob/510b4468d65ecb52759c9c660bf5c4791361d63f/trimesh/intersections.py#L429-L617
# https://github.com/mikedh/trimesh/blob/510b4468d65ecb52759c9c660bf5c4791361d63f/trimesh/constants.py

import logging
import numpy as np

from ..tri import FACE_DTYPE, quads_to_tris

log = logging.getLogger("polliwog.plane._trimesh_intersections")


class ToleranceMesh(object):
    """
    ToleranceMesh objects hold tolerance information about meshes.
    Parameters
    ----------------
    tol.merge : float
      When merging vertices, consider vertices closer than this
      to be the same vertex. Here we use the same value (1e-8)
      as SolidWorks uses, according to their documentation.
    """

    def __init__(self, **kwargs):
        # vertices closer than this should be merged
        self.merge = 1e-8


tol = ToleranceMesh()


def unique_bincount(values, minlength=0):
    """
    For arrays of integers find unique values using bin counting.
    Roughly 10x faster for correct input than np.unique

    Parameters
    --------------
    values : (n,) int
      Values to find unique members of
    minlength : int
      Maximum value that will occur in values (values.max())

    Returns
    ------------
    unique : (m,) int
      Unique values in original array
    inverse : (n,) int, optional
      An array such that unique[inverse] == values
    """
    values = np.asanyarray(values)
    if len(values.shape) != 1 or values.dtype.kind != "i":  # pragma: no cover
        raise ValueError("input must be 1D integers!")

    # count the number of occurrences of each value
    counts = np.bincount(values, minlength=minlength)

    # which bins are occupied at all
    # counts are integers so this works
    unique_bin = counts.astype(bool)

    # which values are unique
    # indexes correspond to original values
    unique = np.where(unique_bin)[0]

    # find the inverse to reconstruct original
    inverse = (np.cumsum(unique_bin) - 1)[values]

    return unique, inverse


def slice_faces_plane(
    vertices,
    faces,
    plane_normal,
    plane_origin,
    face_index=None,
    cached_dots=None,
    return_face_mapping=False,
):
    """
    Slice a mesh (given as a set of faces and vertices) with a plane, returning a
    new mesh (again as a set of faces and vertices) that is the
    portion of the original mesh to the positive normal side of the plane.
    Parameters
    ---------
    vertices : (n, 3) float
        Vertices of source mesh to slice
    faces : (n, 3) int
        Faces of source mesh to slice
    plane_normal : (3,) float
        Normal vector of plane to intersect with mesh
    plane_origin :  (3,) float
        Point on plane to intersect with mesh
    face_index : ((m,) int)
        Indexes of faces to slice. When no mask is provided, the
        default is to slice all faces.
    cached_dots : (n, 3) float
        If an external function has stored dot
        products pass them here to avoid recomputing
    return_face_mapping : bool
        When True, return the indices of the old faces to which the new faces
        correspond.
    Returns
    ----------
    new_vertices : (n, 3) float
        Vertices of sliced mesh
    new_faces : (n, 3) int
        Faces of sliced mesh
    """

    if len(vertices) == 0:
        empty = (vertices, faces)
        if return_face_mapping:
            return (*empty, np.arange(len(faces)))
        else:
            return empty

    # Construct a mask for the faces to slice.
    if face_index is None:
        mask = np.ones(len(faces), dtype=np.bool)
    else:
        mask = np.zeros(len(faces), dtype=np.bool)
        mask[face_index] = True

    if cached_dots is not None:  # pragma: no cover
        dots = cached_dots
    else:
        # dot product of each vertex with the plane normal indexed by face
        # so for each face the dot product of each vertex is a row
        # shape is the same as faces (n,3)
        dots = np.einsum("i,ij->j", plane_normal, (vertices - plane_origin).T)

    # Find vertex orientations w.r.t. faces for all triangles:
    #  -1 -> vertex "inside" plane (positive normal direction)
    #   0 -> vertex on plane
    #   1 -> vertex "outside" plane (negative normal direction)
    signs = np.zeros(len(vertices), dtype=np.int8)
    signs[dots < -tol.merge] = 1
    signs[dots > tol.merge] = -1
    signs = signs[faces]

    # Find all triangles that intersect this plane
    # onedge <- indices of all triangles intersecting the plane
    # inside <- indices of all triangles "inside" the plane (positive normal)
    signs_sum = signs.sum(axis=1, dtype=np.int8)
    signs_asum = np.abs(signs).sum(axis=1, dtype=np.int8)

    # Cases:
    # (0,0,0),  (-1,0,0),  (-1,-1,0), (-1,-1,-1) <- inside
    # (1,0,0),  (1,1,0),   (1,1,1)               <- outside
    # (1,0,-1), (1,-1,-1), (1,1,-1)              <- onedge
    onedge = np.logical_and(
        np.logical_and(signs_asum >= 2, np.abs(signs_sum) <= 1), mask
    )
    inside = np.logical_or((signs_sum == -signs_asum), ~mask)

    # Automatically include all faces that are "inside"
    new_faces = faces[inside]
    if return_face_mapping:
        new_face_mapping = inside.nonzero()[0]

    # Separate faces on the edge into two cases: those which will become
    # quads (two vertices inside plane) and those which will become triangles
    # (one vertex inside plane)
    onedge_quad = np.logical_and(onedge, signs_sum < 0).nonzero()[0]
    cut_faces_quad = faces[onedge_quad]
    cut_signs_quad = signs[onedge_quad]
    onedge_tri = np.logical_and(onedge, signs_sum >= 0).nonzero()[0]
    cut_faces_tri = faces[onedge_tri]
    cut_signs_tri = signs[onedge_tri]

    # If no faces to cut, the surface is not in contact with this plane.
    # Thus, return a mesh with only the inside faces
    if len(cut_faces_quad) + len(cut_faces_tri) == 0:

        if len(new_faces) == 0:
            # if no new faces at all return empty arrays
            empty = (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=FACE_DTYPE),
            )
            if return_face_mapping:
                return (*empty, new_face_mapping)
            else:
                return empty

        # Renumber vertices, dropping any which have been orphaned.
        unique, inverse = unique_bincount(new_faces.ravel())
        final = (
            vertices[unique],
            inverse.reshape((-1, 3)),
        )
        if return_face_mapping:
            return (*final, new_face_mapping)
        else:
            return final

    # Extract the intersections of each triangle's edges with the plane
    o = vertices[faces][onedge]  # origins
    d = np.roll(o, -1, axis=1) - o  # directions
    num = (plane_origin - o).dot(plane_normal)  # compute num/denom
    denom = np.dot(d, plane_normal)
    denom[denom == 0.0] = 1e-12  # prevent division by zero
    dist = np.divide(num, denom)
    # intersection points for each segment
    int_points = np.einsum("ij,ijk->ijk", dist, d) + o

    # Initialize the array of new vertices with the current vertices
    new_vertices = vertices

    # Handle the case where a new quad is formed by the intersection
    # First, extract the intersection points belonging to a new quad
    quad_int_points = int_points[(signs_sum < 0)[onedge], :, :]
    num_quads = len(quad_int_points)
    if num_quads > 0:
        # Extract the vertex on the outside of the plane, then get the vertices
        # (in CCW order of the inside vertices)
        quad_int_inds_col = np.where(cut_signs_quad == 1)[1].reshape(-1, 1)
        quad_int_verts = cut_faces_quad[
            np.tile(np.arange(num_quads).reshape(num_quads, 1), 2),
            (np.repeat(quad_int_inds_col, 2, axis=1) + np.array([1, 2])) % 3,
        ]

        # Fill out new quad faces with the intersection points as vertices
        new_quad_faces = np.append(
            quad_int_verts,
            np.arange(len(new_vertices), len(new_vertices) + 2 * num_quads).reshape(
                num_quads, 2
            ),
            axis=1,
        )

        # Extract correct intersection points from int_points and order them in
        # the same way as they were added to faces
        new_quad_vertices = quad_int_points[
            np.tile(np.arange(num_quads).reshape(num_quads, 1), 2),
            (np.repeat(quad_int_inds_col, 2, axis=1) + np.array([2, 0])) % 3,
            :,
        ].reshape(2 * num_quads, 3)

        # Add new vertices to existing vertices, triangulate quads, and add the
        # resulting triangles to the new faces
        new_vertices = np.append(new_vertices, new_quad_vertices, axis=0)
        new_tri_faces_from_quads = quads_to_tris(new_quad_faces)
        new_faces = np.append(new_faces, new_tri_faces_from_quads, axis=0)
        if return_face_mapping:
            new_face_mapping = np.append(
                new_face_mapping,
                # Two new triangles have been added for each quad.
                np.repeat(onedge_quad, 2),
            )

    # Handle the case where a new triangle is formed by the intersection
    # First, extract the intersection points belonging to a new triangle
    tri_int_points = int_points[(signs_sum >= 0)[onedge], :, :]
    num_tris = len(tri_int_points)
    if num_tris > 0:
        # Extract the single vertex for each triangle inside the plane and get the
        # inside vertices (CCW order)
        tri_int_inds = np.where(cut_signs_tri == -1)[1]
        tri_int_verts = cut_faces_tri[range(num_tris), tri_int_inds].reshape(
            num_tris, 1
        )

        # Fill out new triangles with the intersection points as vertices
        new_tri_faces = np.append(
            tri_int_verts,
            np.arange(len(new_vertices), len(new_vertices) + 2 * num_tris).reshape(
                num_tris, 2
            ),
            axis=1,
        )

        # Extract correct intersection points and order them in the same way as
        # the vertices were added to the faces
        new_tri_vertices = tri_int_points[
            np.tile(np.arange(num_tris).reshape(num_tris, 1), 2),
            (np.repeat(tri_int_inds.reshape(-1, 1), 2, axis=1) + np.array([0, 2])) % 3,
            :,
        ].reshape(2 * num_tris, 3)

        # Append new vertices and new faces
        new_vertices = np.append(new_vertices, new_tri_vertices, axis=0)
        new_faces = np.append(new_faces, new_tri_faces, axis=0)
        if return_face_mapping:
            # One new triangle has been added for each triangle.
            new_face_mapping = np.append(new_face_mapping, onedge_tri)

    # Renumber vertices, dropping any which have been orphaned.
    unique, inverse = unique_bincount(new_faces.ravel())
    final = (
        new_vertices[unique],
        inverse.reshape((-1, 3)),
    )
    if return_face_mapping:
        return (*final, new_face_mapping)
    else:
        return final
