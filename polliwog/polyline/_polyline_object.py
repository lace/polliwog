import numpy as np
from vg.compat import v2 as vg


class Polyline(object):
    """
    Represent the geometry of a polygonal chain in 3-space. The chain may be
    open or closed.

    There are no constraints on the geometry. For example, the chain may be
    simple or self-intersecting, and the points need not be unique.

    The methods do not mutate; they create new polylines which exhibit the
    requested mutation. However, immutability is not enforced. If you wish
    you can mutate a polyline by updating `polyline.v` or `polyline.is_closed`.
    """

    def __init__(self, v, is_closed=False):
        """
        v: np.array containing points in 3-space.
        is_closed: True indicates a closed chain, which has an extra
          segment connecting the last point back to the first
          point.

        """
        # Avoid invoking _update_edges before setting closed and v.
        self.__dict__["is_closed"] = is_closed
        self.v = v

    @classmethod
    def join(cls, *polylines, is_closed=False):
        """
        Join together a stack of open polylines end-to-end into one
        contiguous polyline. The last vertex of the first polyline is
        connected to the first vertex of the second polyline, and so on.
        """
        if len(polylines) == 0:
            raise ValueError("Need at least one polyline to join")
        if any([polyline.is_closed for polyline in polylines]):
            raise ValueError("Expected input polylines to be open, not closed")
        return cls(
            np.vstack([polyline.v for polyline in polylines]), is_closed=is_closed
        )

    def __repr__(self):
        if self.v is not None and self.num_v != 0:
            if self.is_closed:
                return "<closed Polyline with {} verts>".format(self.num_v)
            else:
                return "<open Polyline with {} verts>".format(self.num_v)
        else:
            return "<Polyline with no verts>"

    def __len__(self):
        return self.num_v

    @property
    def num_v(self):
        """
        Return the number of vertices in the polyline.
        """
        return len(self.v)

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
        return self.__class__(v, is_closed=self.is_closed)

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
        num_edges = num_vertices if self.is_closed else num_vertices - 1

        edges = np.vstack([np.arange(num_edges), np.arange(num_edges) + 1]).T

        if self.is_closed:
            edges[-1][1] = 0

        edges.flags.writeable = False

        self.__dict__["e"] = edges

    @property
    def v(self):
        return self.__dict__["v"]

    @v.setter
    def v(self, new_v):
        """
        Update the vertices to a new array-like thing containing points
        in 3D space. Set to None for an empty polyline.

        """
        if new_v is not None:
            vg.shape.check_value(new_v, (-1, 3))
        self.__dict__["v"] = new_v
        self._update_edges()

    @property
    def is_closed(self):
        return self.__dict__["is_closed"]

    @is_closed.setter
    def is_closed(self, new_is_closed):
        """
        Update whether the polyline is closed or open.

        """
        self.__dict__["is_closed"] = new_is_closed
        self._update_edges()

    @property
    def e(self):
        """
        Return the edges of the polyline: an array containing a pair of
        vertex indices for each edge. This is derived automatically from
        `self.v` and `self.is_closed` whenever those values are set.
        """
        return self.__dict__["e"]

    @property
    def segments(self):
        """
        Coordinate pairs for each segment.
        """
        return self.v[self.e]

    @property
    def segment_vectors(self):
        """
        Vectors spanning each segment.
        """
        segments = self.segments
        return segments[:, 1] - segments[:, 0]

    @property
    def segment_lengths(self):
        """
        The length of each of the segments.

        """
        if self.e is None:
            return np.zeros(0)
        else:
            segments = self.segments
            return vg.euclidean_distance(segments[:, 0], segments[:, 1])

    @property
    def total_length(self):
        """
        The total length of all the segments.

        """
        return np.sum(self.segment_lengths)

    @property
    def path_centroid(self):
        """
        The weighted average of all the points along the edges of the polyline.
        """
        edge_centers = np.average(self.segments, axis=1)
        return np.average(edge_centers, weights=self.segment_lengths, axis=0)

    @property
    def bounding_box(self):
        """
        The bounding box which encompasses the entire polyline.
        """
        from .. import Box

        if self.num_v == 0:
            return None

        return Box.from_points(self.v)

    def index_of_vertex(self, point, atol=1e-08):
        """
        Return the index of the vertex with the given point. If there are
        coincident vertices at that point, return the one at the lowest
        index.
        """
        vg.shape.check(locals(), "point", (3,))

        (matching_indices,) = (
            np.isclose(self.v - point, 0, atol=atol).all(axis=1).nonzero()
        )

        try:
            return matching_indices[0]
        except IndexError:
            # `pass` before `raise` to avoid propagating the IndexError.
            pass
        raise ValueError("No matching vertex")

    def with_insertions(self, points, indices, ret_new_indices=False):
        """
        Return a new polyline with the given points inserted before the given
        indices.

        With `ret_new_indices=True`, also returns the new indices of the
        original vertices and the new indices of the inserted points.
        """
        k = vg.shape.check(locals(), "points", (-1, 3))
        vg.shape.check(locals(), "indices", (k,))

        new_polyline = Polyline(
            v=np.insert(self.v, indices, points, axis=0),
            is_closed=self.is_closed,
        )

        if not ret_new_indices:
            return new_polyline

        # Compute indices of original vertices.
        old_num_v = self.num_v
        stepwise_index_offsets = np.zeros(old_num_v, dtype=np.int64)
        stepwise_index_offsets[indices[indices < old_num_v]] = 1
        cumulative_index_offsets = np.cumsum(stepwise_index_offsets)
        indices_of_original_vertices = np.arange(old_num_v) + cumulative_index_offsets

        # Compute indices of inserted points. When more than one point is
        # inserted, this will differ from `indices`.
        # TODO: I think this will cause an IndexError when new points are
        # inserted at the end. `indices + cumulative_index_offsets[indices - 1]`
        # would work instead, but would produce an incorrect result for points
        # inserted at the beginning.
        indices_of_inserted_points = indices + cumulative_index_offsets[indices] - 1

        return new_polyline, indices_of_original_vertices, indices_of_inserted_points

    def flipped(self):
        """
        Flip the polyline from end to end. Return a new polyline.
        """
        return Polyline(v=np.flipud(self.v), is_closed=self.is_closed)

    def aligned_with(self, vector):
        """
        Flip the polyline if necessary, so it's aligned with the given
        vector rather than against it. Works on open polylines and considers
        only the two end vertices.
        """
        if self.is_closed:
            raise ValueError("Can't align a closed polyline")

        vg.shape.check(locals(), "vector", (3,))

        if self.num_v < 2:
            return self

        extent = self.v[-1] - self.v[0]
        projected = vg.project(extent, onto=vector)
        if vg.scale_factor(projected, vector) < 0:
            return self.flipped()
        else:
            return self

    def rolled(self, index, ret_edge_mapping=False):
        """
        Return a new Polyline which reindexes the callee polyline, which much
        be closed, so the vertex with the given index becomes vertex 0.

        ret_edge_mapping: if True, return an array that maps from old edge
            indices to new.
        """
        if not self.is_closed:
            raise ValueError("Can't roll an open polyline")

        result = Polyline(v=np.roll(self.v, -index, axis=0), is_closed=True)

        if ret_edge_mapping:
            edge_mapping = np.roll(np.arange(self.num_v), -index)
            return result, edge_mapping
        else:
            return result

    def subdivided_by_length(
        self, max_length, edges_to_subdivide=None, ret_indices=False
    ):
        """
        Subdivide each line segment longer than `max_length` with
        equal-length segments, such that none of the new segments are longer
        than `max_length`. Returns a new Polyline.

        Args:
            max_length (float): The maximum lenth of a segment.
            edges_to_subdivide (np.arraylike): An optional boolean mask the same
                length as the number of edges. Only the edges marked `True` are
                subdivided. The default is to subdivide all edges longer than
                `max_length`.
            ret_indices (bool): When `True`, also returns the indices of the
                original vertices.
        """
        import itertools
        from ..segment import subdivide_segment

        if edges_to_subdivide is None:
            edges_to_subdivide = np.ones(self.num_e, dtype=np.bool)
        else:
            vg.shape.check(locals(), "edges_to_subdivide", (self.num_e,))

        old_num_e = self.num_e
        old_num_v = self.num_v
        num_segments_needed = np.ceil(self.segment_lengths / max_length).astype(
            dtype=np.int64
        )
        (es_to_subdivide,) = np.logical_and(
            edges_to_subdivide, num_segments_needed > 1
        ).nonzero()
        vs_to_insert = [
            subdivide_segment(
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
        new_polyline = Polyline(
            v=np.concatenate(
                list(
                    itertools.chain(
                        *zip(
                            splits_of_original_vs,
                            vs_to_insert + [np.empty((0, 3), dtype=np.float64)],
                        )
                    )
                )
            ),
            is_closed=self.is_closed,
        )

        if not ret_indices:
            return new_polyline

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
                num_segments_inserted[:-1] if self.is_closed else num_segments_inserted,
            ]
        )
        cumulative_index_offsets = np.sum(
            np.tril(np.broadcast_to(stepwise_index_offsets, (old_num_v, old_num_v))),
            axis=1,
        )
        indices_of_original_vertices = np.arange(old_num_v) + cumulative_index_offsets
        return new_polyline, indices_of_original_vertices

    def with_segments_bisected(self, segment_indices, ret_new_indices=False):
        """
        Return a new polyline with the given segments cut in half.

        With `ret_new_indices=True`, also returns the new indices of the
        original vertices and the new indices of the inserted points.
        """
        return self.with_insertions(
            points=np.mean(self.segments[segment_indices], axis=0),
            indices=self.e[segment_indices][:, 1],
            ret_new_indices=ret_new_indices,
        )

    def apex(self, axis):
        """
        Find the most extreme point in the direction of the axis provided.

        axis: A vector, which is an 3x1 np.array.

        """
        return vg.apex(self.v, axis)

    def intersect_plane(self, plane, ret_edge_indices=False):
        """
        Returns the points of intersection between the plane and any of the
        edges of `polyline`, which should be an instance of Polyline.

        TODO: This doesn't correctly handle vertices which lie on the plane.
        """
        # TODO: Refactor to use `..plane.intersections.intersect_segment_with_plane()`.
        # Identify edges with endpoints that are not on the same side of the plane
        signed_distances = plane.signed_distance(self.v)
        which_es = np.abs(np.sign(signed_distances)[self.e].sum(axis=1)) != 2
        # For the intersecting edges, compute the distance of the endpoints to the plane
        endpoint_distances = np.abs(signed_distances[self.e[which_es]])
        # Normalize the rows of endpoint_distances
        t = endpoint_distances / endpoint_distances.sum(axis=1)[:, np.newaxis]
        # Take a weighted average of the endpoints to obtain the points of intersection
        intersection_points = (
            (1.0 - t[:, :, np.newaxis]) * self.segments[which_es]
        ).sum(axis=1)
        if ret_edge_indices:
            return intersection_points, which_es.nonzero()[0]
        else:
            return intersection_points

    def sliced_by_plane(self, plane):
        """
        Return a new Polyline which keeps only the part that is in front of the given
        plane.

        For open polylines, the plane must intersect the polyline exactly once.

        For closed polylines, the plane must intersect the polyline exactly
        twice, leaving a single contiguous segment in front.
        """
        from ._slice_by_plane import slice_open_polyline_by_plane

        if self.is_closed and self.num_v > 1:
            signed_distances = plane.signed_distance(self.v)
            signs_of_verts = np.sign(signed_distances)
            # For closed polylines, roll the edges so the ones in front of the
            # plane start at index 1 and the one to be cut is at index 0. (If
            # that edge stops directly on the plane, it may not actually need
            # to be cut.) This reduces it to the open polyline intersection
            # problem.
            if signs_of_verts[-1] == 1:
                # e.g. signs_of_verts = np.array([1, -1, -1, 1, 1, 1, 1])
                (vertices_not_in_front,) = np.where(signs_of_verts != 1)
                roll = -vertices_not_in_front[-1]
            else:
                # e.g. signs_of_verts = np.array([-1, 1, 1, 1, 1, 1, -1, -1])
                (vertices_in_front,) = np.where(signs_of_verts == 1)
                if len(vertices_in_front) > 0:
                    roll = -vertices_in_front[0] + 1
                else:
                    roll = 0
            working_v = np.roll(self.v, roll, axis=0)
        else:
            working_v = self.v

        return Polyline(
            v=slice_open_polyline_by_plane(working_v, plane), is_closed=False
        )

    def sliced_at_indices(self, start, stop):
        """
        Take an slice of the given polyline starting at the `start` vertex
        index and ending just befeor reaching the `stop` vertex index. Always
        returns an open polyline.

        When called on a closed polyline, the indies can wrap around the end.
        """
        if stop <= start:
            if self.is_closed:
                num_to_keep = len(self.v) - start + stop
                working_v = np.roll(self.v, -start, axis=0)[0:num_to_keep]
            else:
                raise ValueError(
                    "For an open polyline, start index of slice should be less than stop index"
                )
        else:
            working_v = self.v[start:stop]
        return Polyline(v=working_v, is_closed=False)

    def nearest(self, points, ret_segment_indices=False):
        """
        For the given query point or points, return the nearest point on the
        polyline. With `ret_segment_indices=True`, also return the segment
        indices of those points.
        """
        from .._common.shape import columnize
        from ..segment import closest_point_of_line_segment

        points, _, transform_result = columnize(points, name="points")
        num_points = len(points)

        stacked_points = np.repeat(points, self.num_e, axis=0)
        closest_points_of_segments = closest_point_of_line_segment(
            points=stacked_points,
            start_points=np.tile(self.segments[:, 0], (num_points, 1)),
            segment_vectors=np.tile(self.segment_vectors, (num_points, 1)),
        )
        distance_to_closest_points_of_segments = vg.euclidean_distance(
            stacked_points, closest_points_of_segments
        )

        closest_points_of_segments = closest_points_of_segments.reshape(
            num_points, self.num_e, 3
        )
        distance_to_closest_points_of_segments = (
            distance_to_closest_points_of_segments.reshape(num_points, self.num_e)
        )

        indices_of_nearest_segments = np.argmin(
            distance_to_closest_points_of_segments, axis=1
        )
        closest_points_of_polyline = np.take_along_axis(
            closest_points_of_segments,
            indices_of_nearest_segments.reshape(num_points, 1, 1),
            axis=1,
        ).reshape(num_points, 3)

        if ret_segment_indices:
            return (
                transform_result(closest_points_of_polyline),
                transform_result(indices_of_nearest_segments),
            )
        else:
            return transform_result(closest_points_of_polyline)

    def sliced_at_points(self, start_point, end_point, atol=1e-8):
        """
        Take a slice of the given polyline at the given start and end points.
        These are expected to be on a vertex or on a segment. If on a segment
        (or near to but not directly on a segment) a new point is inserted
        at exactly the given point.
        """
        vg.shape.check(locals(), "start_point", (3,))
        vg.shape.check(locals(), "end_point", (3,))

        working_polyline = self

        try:
            # Check if the start point intersects a vertex. If it does, great;
            # if not, insert it.
            start_v_index = working_polyline.index_of_vertex(start_point)
        except ValueError:
            nearest_point, segment_index = working_polyline.nearest(
                start_point, ret_segment_indices=True
            )
            (_, start_v_index) = working_polyline.e[segment_index]
            working_polyline = working_polyline.with_insertions(
                points=nearest_point.reshape(-1, 3),
                indices=np.array([start_v_index]),
            )

        try:
            end_v_index = working_polyline.index_of_vertex(end_point)
        except ValueError:
            nearest_point, segment_index = working_polyline.nearest(
                end_point, ret_segment_indices=True
            )
            (_, end_v_index) = working_polyline.e[segment_index]
            (
                working_polyline,
                indices_of_original_vertices,
                _,
            ) = working_polyline.with_insertions(
                points=nearest_point.reshape(-1, 3),
                indices=np.array([end_v_index]),
                ret_new_indices=True,
            )
            start_v_index = indices_of_original_vertices[start_v_index]

        # Then slice at those points.
        return working_polyline.sliced_at_indices(start_v_index, end_v_index + 1)

    def sectioned(self, section_breakpoints, copy_vs=False):
        """
        Section the given open polyline at the given breakpoints, which indicate
        where one segment ends and the next one starts. Each of the breakpoint
        vertices is included as an endpoint in one section and a start point in
        the next section.

        Args:
            breakpoints (np.arraylike): The indices of the breakpoints.
            copy_vs (bool): When `True`, copy the vertices into the new polylines.
                When `False`, return polylines with views for vertex arrays.

        Returns:
            list: A list of the sectioned polylines.
        """
        if self.is_closed:
            raise NotImplementedError("Not implemented for closed polylines")

        vg.shape.check(locals(), "section_breakpoints", (-1,))

        section_breakpoints = section_breakpoints.astype(np.int64)
        maybe_copy = np.copy if copy_vs else lambda vs: vs
        section_starts = np.hstack([np.array(0, dtype=np.int64), section_breakpoints])
        section_ends = np.hstack(
            [section_breakpoints + 1, np.array([self.num_v], dtype=np.int64)]
        )

        edges_per_section = section_ends - section_starts - 1
        if (edges_per_section < 1).any():
            raise ValueError("Every section must have at least one edge")

        return [
            Polyline(v=maybe_copy(self.v[start:end]), is_closed=False)
            for (start, end) in zip(section_starts, section_ends)
        ]

    def point_along_path(self, fraction_of_total):
        """
        Selects a point the given fraction of the total length of the polyline. For
        example, to find the halfway point, pass `fraction_of_total=0.5`. Also works
        with stacked values, e.g. `fraction_of_total=np.linspace(0, 1, 11)`.

        Args:
            fraction_of_total (object): Fraction of the total length, from 0 to 1

        Returns (object):
            A point on the polyline that is the given fraction of the total length
            from the starting point to the endpoint. For stacked fractions, return
            the points.
        """
        from .._common.shape import columnize

        fraction_of_total, _, transform_result = columnize(
            fraction_of_total, (-1,), name="fraction_of_total"
        )

        if np.any(0 > fraction_of_total) or np.any(fraction_of_total > 1):
            raise ValueError("fraction_of_total must be a value between 0 and 1")

        desired_length = self.total_length * fraction_of_total
        cumulative_length = np.cumsum([0, *self.segment_lengths])
        index_of_segment = (
            np.argmax(cumulative_length.reshape(-1, 1) > desired_length, axis=0) - 1
        )

        return transform_result(
            self.v[index_of_segment]
            + (desired_length - cumulative_length[index_of_segment]).reshape(-1, 1)
            * vg.normalize(self.segment_vectors[index_of_segment])
        )
