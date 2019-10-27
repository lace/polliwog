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
    def join(cls, *polylines):
        """

        Join together a stack of open polylines end-to-end into one
        contiguous polyline. The last vertex of the first polyline is
        connected to the first vertex of the second polyline, and so on.
        """
        if len(polylines) == 0:
            raise ValueError("Need at least one polyline to join")
        if any([polyline.is_closed for polyline in polylines]):
            raise ValueError("Expected input polylines to be open, not closed")
        return cls(np.vstack([polyline.v for polyline in polylines]), is_closed=False)

    def __repr__(self):
        if self.v is not None and len(self.v) != 0:
            if self.is_closed:
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
    def is_closed(self, val):
        """
        Update whether the polyline is closed or open.

        """
        self.__dict__["is_closed"] = val
        self._update_edges()

    @property
    def e(self):
        """
        Return a np.array of edges. Derived automatically from self.v
        and self.is_closed whenever those values are set.

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

    @property
    def bounding_box(self):
        """
        The bounding box which encompasses the entire polyline.
        """
        from ..box.box import Box

        if self.num_v == 0:
            return None

        return Box.from_points(self.v)

    def flip(self):
        """
        Flip the polyline from end to end.
        """
        self.v = np.flipud(self.v)

        return self

    def oriented_along(self, along, reverse=False):
        """
        Flip the polyline, if necessary, so that it points (approximately)
        along the second vector rather than (approximately) opposite it.
        """
        if self.is_closed:
            raise ValueError("Can't reorient a closed polyline")

        vg.shape.check(locals(), "along", (3,))

        if self.num_v < 2:
            return self

        extent = self.v[-1] - self.v[0]
        projected = vg.project(extent, onto=along)
        if vg.scale_factor(projected, along) * (-1 if reverse else 1) < 0:
            return self.copy().flip()
        else:
            return self

    def reindexed(self, index, ret_edge_mapping=False):
        """
        Return a new Polyline which reindexes the callee polyline, which much
        be closed, so the vertex with the given index becomes vertex 0.

        ret_edge_mapping: if True, return an array that maps from old edge
            indices to new.
        """
        if not self.is_closed:
            raise ValueError("Can't reindex an open polyline")

        result = Polyline(
            v=np.append(self.v[index:], self.v[0:index], axis=0), is_closed=True
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
                    if self.is_closed
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

    def cut_by_plane(self, plane):
        """
        Return a new Polyline which keeps only the part that is in front of the given
        plane.

        For open polylines, the plane must intersect the polyline exactly once.

        For closed polylines, the plane must intersect the polylint exactly
        twice, leaving a single contiguous segment in front.
        """
        from .cut_by_plane import cut_open_polyline_by_plane

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
                vertices_not_in_front, = np.where(signs_of_verts != 1)
                roll = -vertices_not_in_front[-1]
            else:
                # e.g. signs_of_verts = np.array([-1, 1, 1, 1, 1, 1, -1, -1])
                vertices_in_front, = np.where(signs_of_verts == 1)
                if len(vertices_in_front) > 0:
                    roll = -vertices_in_front[0] + 1
                else:
                    roll = 0
            working_v = np.roll(self.v, roll, axis=0)
        else:
            working_v = self.v

        new_v = cut_open_polyline_by_plane(working_v, plane)
        return Polyline(v=new_v, is_closed=False)
