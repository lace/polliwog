import numpy as np
import vg
from .._temporary.decorators import setter_property


class Polyline(object):
    """
    Represent the geometry of a polygonal chain in 3-space. The
    chain may be open or closed, and there are no constraints on the
    geometry. For example, the chain may be simple or
    self-intersecting, and the points need not be unique.

    Mutable by setting polyline.v or polyline.closed or calling
    a method like polyline.partition_by_length().

    Note this class is distinct from lace.lines.Lines, which
    allows arbitrary edges and enables visualization. To convert to
    a Lines object, use the as_lines() method.

    """

    def __init__(self, v, closed=False):
        """
        v: np.array containing points in 3-space.
        closed: True indicates a closed chain, which has an extra
          segment connecting the last point back to the first
          point.

        """
        # Avoid invoking _update_edges before setting closed and v.
        self.__dict__["closed"] = closed
        self.v = v

    def __repr__(self):
        if self.v is not None and len(self.v) != 0:
            if self.closed:
                return "<closed Polyline with {} verts>".format(len(self))
            else:
                return "<open Polyline with {} verts>".format(len(self))
        else:
            return "<Polyline with no verts>"

    def __len__(self):
        return len(self.v)

    @property
    def num_v(self):
        """
        Return the number of vertices in the polyline.
        """
        return len(self)

    @property
    def num_e(self):
        """
        Return the number of segments in the polyline.
        """
        return len(self.e)

    def copy(self):
        """
        Return a copy of this polyline.

        """
        v = None if self.v is None else np.copy(self.v)
        return self.__class__(v, closed=self.closed)

    def to_dict(self, decimals=3):
        return {
            "vertices": [np.around(v, decimals=decimals).tolist() for v in self.v],
            "edges": self.e,
        }

    def _update_edges(self):
        if self.v is None:
            self.__dict__["e"] = None
            return

        num_vertices = self.v.shape[0]
        num_edges = num_vertices if self.closed else num_vertices - 1

        edges = np.vstack([np.arange(num_edges), np.arange(num_edges) + 1]).T

        if self.closed:
            edges[-1][1] = 0

        edges.flags.writeable = False

        self.__dict__["e"] = edges

    @setter_property
    def v(
        self, val
    ):  # setter_property incorrectly triggers method-hidden. pylint: disable=method-hidden
        """
        Update the vertices to a new array-like thing containing points
        in 3D space. Set to None for an empty polyline.

        """
        if val is not None:
            vg.shape.check_value(val, (-1, 3))
        self.__dict__["v"] = val
        self._update_edges()

    @setter_property
    def closed(self, val):
        """
        Update whether the polyline is closed or open.

        """
        self.__dict__["closed"] = val
        self._update_edges()

    @property
    def e(self):
        """
        Return a np.array of edges. Derived automatically from self.v
        and self.closed whenever those values are set.

        """
        return self.__dict__["e"]

    @property
    def segments(self):
        """
        Coordinate pairs for each segment.
        """
        return self.v[self.e]

    @property
    def segment_lengths(self):
        """
        The length of each of the segments.

        """
        if self.e is None:
            return np.zeros(0)
        else:
            v1s = self.v[self.e[:, 0]]
            v2s = self.v[self.e[:, 1]]
            return vg.euclidean_distance(v1s, v2s)

    @property
    def total_length(self):
        """
        The total length of all the segments.

        """
        return np.sum(self.segment_lengths)

    @property
    def segment_vectors(self):
        """
        Vectors spanning each segment.
        """
        segments = self.segments
        return segments[:, 1] - segments[:, 0]

    def flip(self):
        """
        Flip the polyline from end to end.
        """
        self.v = np.flipud(self.v)

    def reindexed(self, index, ret_edge_mapping=False):
        """
        Return a new Polyline which reindexes the callee polyline, which much
        be closed, so the vertex with the given index becomes vertex 0.

        ret_edge_mapping: if True, return an array that maps from old edge
            indices to new.
        """
        if not self.closed:
            raise ValueError("Can't reindex an open polyline")

        result = Polyline(
            v=np.append(self.v[index:], self.v[0:index], axis=0), closed=True
        )

        if ret_edge_mapping:
            edge_mapping = np.append(
                np.arange(index, len(self.v)), np.arange(0, index), axis=0
            )
            return result, edge_mapping
        else:
            return result

    def partition_by_length(self, max_length, ret_indices=False):
        """
        Subdivide each line segment longer than max_length with
        equal-length segments, such that none of the new segments
        are longer than max_length.

        ret_indices: If True, return the indices of the original vertices.
          Otherwise return self for chaining.

        """
        import itertools
        from ..segment.segment import partition_segment
        from ..plane.intersections import intersect_segment_with_plane

        old_num_e = self.num_e
        old_num_v = self.num_v
        num_segments_needed = np.ceil(self.segment_lengths / max_length).astype(
            dtype=np.int64
        )
        es_to_subdivide, = (num_segments_needed > 1).nonzero()
        vs_to_insert = [
            partition_segment(
                self.v[self.e[old_e_index][0]],
                self.v[self.e[old_e_index][1]],
                np.int(num_segments_needed[old_e_index]),
                endpoint=False,
            )[
                # Exclude the start point, which like the endpoint, is already
                # present.
                1:
            ]
            for old_e_index in es_to_subdivide
        ]

        splits_of_original_vs = np.vsplit(self.v, es_to_subdivide + 1)
        self.v = np.concatenate(
            list(
                itertools.chain(
                    *zip(
                        splits_of_original_vs,
                        vs_to_insert + [np.empty((0, 3), dtype=np.float64)],
                    )
                )
            )
        )

        if ret_indices:
            # In a degenerate case, `partition_segment()` may return fewer than
            # the requested number of indices. So, recompute the actual number of
            # segments inserted.
            num_segments_inserted = np.zeros(old_num_e, dtype=np.int64)
            num_segments_inserted[es_to_subdivide] = [len(vs) for vs in vs_to_insert]
            stepwise_index_offsets = np.concatenate(
                [
                    # The first vertex is never moved.
                    np.zeros(1, dtype=np.int64),
                    # In a closed polyline, the last edge goes back to vertex
                    # 0. Subdivisions of that segment do not affect indexing of
                    # any of the vertices (since the original end vertex is
                    # still at index 0).
                    num_segments_inserted[:-1]
                    if self.closed
                    else num_segments_inserted,
                ]
            )
            cumulative_index_offsets = np.sum(
                np.tril(
                    np.broadcast_to(stepwise_index_offsets, (old_num_v, old_num_v))
                ),
                axis=1,
            )
            return np.arange(old_num_v) + cumulative_index_offsets
        else:
            return self

    def bisect_edges(self, edges):
        """
        Cutting the given edges in half.

        Return an arrray that gives the new indices of the original vertices.
        """
        new_vs = self.v
        indices_of_original_vertices = np.arange(len(self.v))
        for offset, edge_to_subdivide in enumerate(edges):
            new_v = np.mean(self.segments[edge_to_subdivide], axis=0).reshape(-1, 3)
            old_v2_index = self.e[edge_to_subdivide][0] + 1
            insert_at = offset + old_v2_index
            new_vs = np.insert(new_vs, insert_at, new_v, axis=0)
            indices_of_original_vertices[old_v2_index:] = (
                indices_of_original_vertices[old_v2_index:] + 1
            )
        self.v = new_vs
        return indices_of_original_vertices

    def apex(self, axis):
        """
        Find the most extreme point in the direction of the axis provided.

        axis: A vector, which is an 3x1 np.array.

        """
        return vg.apex(self.v, axis)

    def intersect_plane(self, plane, ret_edge_and_vertex_indices=False):
        """
        Returns the points of intersection between the plane and any of the
        edges of `polyline`, which should be an instance of Polyline.
        """
        if self.num_v == 0:
            intersection_points = np.zeros((3, 0))
            if ret_edge_and_vertex_indices:
                edge_indices = np.zeros((3,))
                vertex_indices = np.zeros((3,))
                return intersection_points, vertex_indices, edge_indices
            else:
                return intersection_points

        # Identify edges with endpoints that are not on the same side of the plane
        signed_distances = plane.signed_distance(self.v)
        signs_of_verts = np.sign(signed_distances)

        # First handle the edge case where a vertex lies on the plane. Because
        # the goal is to produce the correct result when a vertex lies on the
        # plane, there is no need to detect using a floating-point tolerance.
        # It is perfectly acceptable to miss an intersection which lies
        # very close to one side or the other. So long as the input polyline is
        # well-formed (i.e. not zig-zagging near the plane), the result will be
        # correct.
        verts_of_edges = self.v[self.e]
        degenerate_segments = np.all(
            verts_of_edges[:, 0] == verts_of_edges[:, 1], axis=1
        )

        signs_of_verts_by_edge = signs_of_verts[self.e]
        edges_in_plane = np.all(signs_of_verts_by_edge == 0, axis=1)
        # nondegenerate_edges_in_plane = np.logical_and(
        #     edges_in_plane, ~degenerate_segments
        # )
        # TODO When there are contiguous nondegenerate edges which lie the plane,
        # an excpetion is raised. This case could be handled better by collapsing
        # them into a single vertex.

        edges_crossing_plane = (
            signs_of_verts_by_edge[:, 0] * signs_of_verts_by_edge[:, 1] == -1
        )

        es_to_cut = edges_crossing_plane
        es_to_drop = edges_in_plane
        es_to_preserve = np.logical_and(~es_to_cut, ~es_to_drop)

        # vs_to_drop = either end of es_to_drop, and unwanted end of es_to_cut.
        # vs to drop: the ones with sign 0 or -1.
        vs_to_drop = signs_of_verts < 1
        num_changes = find_changes(vs_to_drop, wrap=self.closed)
        if self.closed and num_changes != 2:
            raise ValueError("")

        # Check integrity. All the verts of edges being dropped should be contiguous.

        num_v_intersections = np.sum(signs == 0, axis=1)
        intersecting_vertex_indices, = sides_of_edge_vertices.sum(axis=1) == 1
        intersecting_edge_indices, = np.abs(sides_of_edge_vertices.sum(axis=1)) == 2
        # For the intersecting edges, compute the distance of the endpoints to the plane
        endpoint_distances = np.abs(signed_distances[self.e[intersecting_edge_indices]])
        # Normalize the rows of endpoint_distances
        t = endpoint_distances / endpoint_distances.sum(axis=1)[:, np.newaxis]
        # Take a weighted average of the endpoints to obtain the points of intersection
        intersection_points = (
            (1.0 - t[:, :, np.newaxis]) * self.segments[which_es]
        ).sum(axis=1)
        if ret_edge_and_vertex_indices:
            edge_indices, = which_es.nonzero()
            return intersection_points, edge_indices
        else:
            return intersection_points

    def cut_by_plane(self, plane):
        """
        Return a new Polyline which keeps only the part that is in front of the given
        plane.

        For open polylines, the plane must intersect the polyline exactly once.

        For closed polylines, the plane must intersect the polylint exactly
        twice, leaving a single contiguous segment in front.
        """
        from .cut_by_plane import cut_open_polyline_by_plane
        from ._array import find_changes

        if self.num_v == 0:
            raise ValueError("A plane can't intersect a polyline with no points")

        signed_distances = plane.signed_distance(self.v)
        signs_of_verts = np.sign(signed_distances)

        signs_of_verts_by_edge = signs_of_verts[self.e]

        if self.closed:
            # Handle an exception case that will cause a crash here.
            if self.num_v == 1:
                if signs_of_verts[0] == 0:
                    return Polyline(v=self.v, closed=False)
                else:
                    raise ValueError("Plane does not intersect polyline")

            # For closed polylines, roll the edges so the ones in front of the
            # plane start at index 1 and the one to be cut is at index 0. (If
            # that edge stops directly on the plane, it may not actually need
            # to be cut.)
            if signs_of_verts[-1] == 1:
                # e.g. signs_of_verts = np.array([1, -1, -1, 1, 1, 1, 1])
                vertices_not_in_front, = np.where(signs_of_verts != 1)
                roll = -vertices_not_in_front[-1]
            else:
                # e.g. signs_of_verts = np.array([-1, 1, 1, 1, 1, 1, -1, -1])
                vertices_in_front, = np.where(signs_of_verts == 1)
                if len(vertices_in_front) > 0:
                    roll = -vertices_in_front[0] + 1
                else:
                    # This is the extreme case of a polyline which intersects
                    # only at points and/or edges in the plane.
                    # e.g. signs_of_verts = np.array([-1, -1, -1, 0, 0, -1, -1])
                    vertices_in_plane, = np.where(signs_of_verts == 0)
                    roll = -vertices_in_plane[0] + 1
            working_v = np.roll(self.v, roll, axis=0)
            signs_of_working_v = np.roll(signs_of_verts, roll)
            # Assertions.
            np.testing.assert_array_equal(signs_of_working_v, plane.sign(working_v))
            if np.max(signs_of_verts) == 0:
                assert plane.signed_distance(working_v)[1] == 0
            else:
                assert plane.sign(working_v)[0] < 1
                assert plane.sign(working_v)[1] == 1
                assert (plane.sign(working_v)[2:] < 1).nonzero()[0][0] + 2
        else:
            working_v = self.v

        new_v = cut_open_polyline_by_plane(working_v, plane)
        return Polyline(v=new_v, closed=False)

        # import pdb

        # pdb.set_trace()
        # return

        # # First, remove all the segments that lie entirely in the plane. These
        # # could either be degenerate segments or
        # # edges_in_plane = np.all(signs_of_verts_by_edge == 0, axis=1)
        # # verts_on_
        # # new_vs =

        # # If it's open, compute the cut segment and append or prepend it.
        # if self.closed:
        #     # Reindex the polyline so it starts with a contiguous subchain which
        #     # lies in front of the plane. (In the extreme cases of a polyline
        #     # which intersects a plane at a single point, it starts at that point.)
        #     first_point_is_in_back = signs_of_verts[0] == -1
        #     if first_point_is_in_back:
        #         # e.g. v_sign = np.array([-1, 1, 1, 1, -1, -1, -1])
        #         points_on_or_in_front, = np.where(signs_of_verts >= 0)
        #         working = self.reindexed(index=points_on_or_in_front[0])
        #     else:
        #         # e.g. v_sign = np.array([1, -1, -1, -1, 1, 1, 1])
        #         points_in_back, = np.where(signs_of_verts < 0)
        #         working = self.reindexed(index=points_in_back[-1] + 1)
        # else:
        #     intersecting_edge_index, = np.nonzero(changes)[0]

        # intersecting_edge = self.e[intersecting_edge_index]
        # intersecting_segment_vector = self.segment_vectors[intersecting_edge_index]
        # # Do we want the edges before or after the intersecting edge? Determine that
        # # by checking whether the intersecting edge crosses from back to front or
        # # front to back. If back to front, keep the edges after it. If front to back,
        # # keep the edges before it. In either case, add a new edge for the portion of
        # # the intersecting edge that is in front.
        # #
        # # In case a vert of the intersecting edge lies on the plane, use a
        # # vector to identify which direction it's facing.
        # if vg.scalar_projection(intersecting_segment_vector, onto=plane.normal) > 0:
        #     new_v = np.vstack([intersection_point, self.v[intersecting_edge[1] :]])
        # else:
        #     new_v = np.vstack([self.v[: intersecting_edge[0] + 1], intersection_point])
        # return Polyline(v=new_v, closed=False)

        # # If it's closed, rotate so it starts with the first kept segment, then
        # # compute the two cut segments.

        # edges_to_cut = signs_of_verts_by_edge[:, 0] * signs_of_verts_by_edge[:, 1] == -1
        # num_edges_to_cut = len(edges_to_cut)
        # new_vertices_for_cut_edges = intersect_segment_with_plane(
        #     start_points=self.v[edges_to_cut],
        #     segment_vectors=self.segment_vectors[edges_to_cut],
        #     points_on_plane=np.broadcast_to(
        #         self.reference_point, (num_edges_to_cut, 3)
        #     ),
        #     plane_normals=np.broadcast_to(self.normal, (num_edges_to_cut, 3)),
        # )

        # edges_to_keep_or_cut = ~np.all(signs_of_verts_by_edge == 0, axis=1)

        # es_to_drop = edges_in_plane
        # es_to_preserve = np.logical_and(~es_to_cut, ~es_to_drop)

        # # First handle the edge case where a vertex lies on the plane. Because
        # # the goal is to produce the correct result when a vertex lies on the
        # # plane, there is no need to detect using a floating-point tolerance.
        # # It is perfectly acceptable to miss an intersection which lies
        # # very close to one side or the other. So long as the input polyline is
        # # well-formed (i.e. not zig-zagging near the plane), the result will be
        # # correct.
        # verts_of_edges = self.v[self.e]
        # degenerate_segments = np.all(
        #     verts_of_edges[:, 0] == verts_of_edges[:, 1], axis=1
        # )

        # signs_of_verts_by_edge = signs_of_verts[self.e]
        # edges_in_plane = np.all(signs_of_verts_by_edge == 0, axis=1)
        # # nondegenerate_edges_in_plane = np.logical_and(
        # #     edges_in_plane, ~degenerate_segments
        # # )
        # # TODO When there are contiguous nondegenerate edges which lie the plane,
        # # an excpetion is raised. This case could be handled better by collapsing
        # # them into a single vertex.

        # edges_crossing_plane = (
        #     signs_of_verts_by_edge[:, 0] * signs_of_verts_by_edge[:, 1] == -1
        # )

        # es_to_cut = edges_crossing_plane
        # es_to_drop = edges_in_plane
        # es_to_preserve = np.logical_and(~es_to_cut, ~es_to_drop)

        # intersection_points, edge_indices = self.intersect_plane(
        #     plane, ret_edge_and_vertex_indices=True
        # )

        # num_edge_indices = len(edge_indices)
        # if num_edge_indices == 0:
        #     raise ValueError("Plane does not intersect polyline")

        # if self.closed:
        #     if num_edge_indices != 2:
        #         raise ValueError(
        #             "Plane intersects polyline at {} points; expected 2".format(
        #                 num_edge_indices
        #             )
        #         )
        #     return self._cut_by_plane_closed(plane)
        # else:
        #     if num_edge_indices > 1:
        #         raise ValueError(
        #             "Plane intersects polyline at {} points; expected 1".format(
        #                 num_edge_indices
        #             )
        #         )
        #     intersection_point = intersection_points[0]
        #     intersecting_edge_index = edge_indices[0]
        #     return self._cut_by_plane_open(
        #         plane, intersection_point, intersecting_edge_index
        #     )
