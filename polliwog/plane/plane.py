import numpy as np
from blmath.numerics import vx

class Plane(object):
    '''
    A 2-D plane in 3-space (not a hyperplane).

    Params:
        - point_on_plane, plane_normal:
            1 x 3 np.arrays
    '''

    def __init__(self, point_on_plane, unit_normal):
        if vx.almost_zero(unit_normal):
            raise ValueError('unit_normal should not be the zero vector')

        unit_normal = vx.normalize(unit_normal)

        self._r0 = np.asarray(point_on_plane)
        self._n = np.asarray(unit_normal)

    def __repr__(self):
        return "<Plane of {} through {}>".format(self.normal, self.reference_point)

    @classmethod
    def from_points(cls, p1, p2, p3):
        '''
        If the points are oriented in a counterclockwise direction, the plane's
        normal extends towards you.

        '''
        from blmath.numerics import as_numeric_array

        p1 = as_numeric_array(p1, shape=(3,))
        p2 = as_numeric_array(p2, shape=(3,))
        p3 = as_numeric_array(p3, shape=(3,))

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        return cls(point_on_plane=p1, unit_normal=normal)

    @classmethod
    def from_points_and_vector(cls, p1, p2, vector):
        '''
        Compute a plane which contains two given points and the given
        vector. Its reference point will be p1.

        For example, to find the vertical plane that passes through
        two landmarks:

            from_points_and_normal(p1, p2, vector)

        Another way to think about this: identify the plane to which
        your result plane should be perpendicular, and specify vector
        as its normal vector.

        '''
        from blmath.numerics import as_numeric_array

        p1 = as_numeric_array(p1, shape=(3,))
        p2 = as_numeric_array(p2, shape=(3,))

        v1 = p2 - p1
        v2 = as_numeric_array(vector, shape=(3,))
        normal = np.cross(v1, v2)

        return cls(point_on_plane=p1, unit_normal=normal)

    @classmethod
    def fit_from_points(cls, points):
        '''
        Fits a plane whose normal is orthgonal to the first two principal axes
        of variation in the data and centered on the points' centroid.
        '''
        eigval, eigvec = np.linalg.eig(np.cov(points.T))
        ordering = np.argsort(eigval)[::-1]
        normal = np.cross(eigvec[:, ordering[0]], eigvec[:, ordering[1]])
        return cls(points.mean(axis=0), normal)

    @property
    def equation(self):
        '''
        Returns parameters A, B, C, D as a 1 x 4 np.array, where

            Ax + By + Cz + D = 0

        defines the plane.

        params:
            - normalized:
                Boolean, indicates whether or not the norm of the vector [A, B, C] is 1.
                Useful when computing the distance from a point to the plane.
        '''
        A, B, C = self._n
        D = -self._r0.dot(self._n)

        return np.array([A, B, C, D])

    @property
    def reference_point(self):
        '''
        The point used to create this plane.

        '''
        return self._r0

    @property
    def canonical_point(self):
        '''
        A canonical point on the plane, the one at which the normal
        would intersect the plane if drawn from the origin (0, 0, 0).

        This is computed by projecting the reference point onto the
        normal.

        This is useful for partitioning the space between two planes,
        as we do when searching for planar cross sections.

        '''
        return self._r0.dot(self._n) * self._n

    @property
    def normal(self):
        '''
        Return the plane's normal vector.

        '''
        return self._n

    def flipped(self):
        '''
        Creates a new Plane with an inverted orientation.
        '''
        normal = self._n * -1
        return Plane(self._r0, normal)

    def sign(self, points):
        '''
        Given an array of points, return an array with +1 for points in front
        of the plane (in the direction of the normal), -1 for points behind
        the plane (away from the normal), and 0 for points on the plane.

        '''
        return np.sign(self.signed_distance(points))

    def points_in_front(self, points, inverted=False, ret_indices=False):
        '''
        Given an array of points, return the points which lie either on the
        plane or in the half-space in front of it (i.e. in the direction of
        the plane normal).

        points: An array of points.
        inverted: When `True`, invert the logic. Return the points that lie
          behind the plane instead.
        ret_indices: When `True`, return the indices instead of the points
          themselves.

        '''
        sign = self.sign(points)

        if inverted:
            mask = np.less_equal(sign, 0)
        else:
            mask = np.greater_equal(sign, 0)

        indices = np.flatnonzero(mask)

        return indices if ret_indices else points[indices]

    def signed_distance(self, points):
        '''
        Returns the signed distances given an np.array of 3-vectors.

        Params:
            - points:
                V x 3 np.array
        '''
        return np.dot(points, self.equation[:3]) + self.equation[3]

    def distance(self, points):
        return np.absolute(self.signed_distance(points))

    def project_point(self, point):
        '''
        Project a given point to the plane.

        '''
        # Translate the point back to the plane along the normal.
        signed_distance_to_point = self.signed_distance(point.reshape((-1, 3)))[0]
        return point - signed_distance_to_point * self._n

    def polyline_xsection(self, polyline, ret_edge_indices=False):
        '''
        Deprecated.
        '''
        return polyline.intersect_plane(self, ret_edge_indices=ret_edge_indices)

    def line_xsection(self, pt, ray):
        return self._line_xsection(np.asarray(pt).ravel(), np.asarray(ray).ravel())

    def _line_xsection(self, pt, ray):
        denom = np.dot(ray, self.normal)
        if denom == 0:
            return None # parallel, either coplanar or non-intersecting
        p = np.dot(self.reference_point - pt, self.normal) / denom
        return p * ray + pt

    def line_segment_xsection(self, a, b):
        return self._line_segment_xsection(np.asarray(a).ravel(), np.asarray(b).ravel())

    def _line_segment_xsection(self, a, b):
        pt = self._line_xsection(a, b-a)
        if pt is not None:
            if any(np.logical_and(pt > a, pt > b)) or any(np.logical_and(pt < a, pt < b)):
                return None
        return pt

    def line_xsections(self, pts, rays):
        denoms = np.dot(rays, self.normal)
        denom_is_zero = denoms == 0
        denoms[denom_is_zero] = np.nan
        p = np.dot(self.reference_point - pts, self.normal) / denoms
        return np.vstack([p, p, p]).T * rays + pts, ~denom_is_zero

    def line_segment_xsections(self, a, b):
        pts, pt_is_valid = self.line_xsections(a, b-a)
        pt_is_out_of_bounds = np.logical_or(np.any(np.logical_and(pts[pt_is_valid] > a[pt_is_valid], pts[pt_is_valid] > b[pt_is_valid]), axis=1),
                                            np.any(np.logical_and(pts[pt_is_valid] < a[pt_is_valid], pts[pt_is_valid] < b[pt_is_valid]), axis=1))
        pt_is_valid[pt_is_valid] = ~pt_is_out_of_bounds
        pts[~pt_is_valid] = np.nan
        return pts, pt_is_valid


    def mesh_xsection(self, m, neighborhood=None):
        '''
        Backwards compatible.
        Returns one polyline that may connect supposedly disconnected components.
        Returns an empty Polyline if there's no intersection.
        '''
        from blmath.geometry import Polyline

        components = self.mesh_xsections(m, neighborhood)
        if len(components) == 0:
            return Polyline(None)
        return Polyline(np.vstack([x.v for x in components]), closed=True)

    def mesh_intersecting_faces(self, m):
        sgn_dists = self.signed_distance(m.v)
        which_fs = np.abs(np.sign(sgn_dists)[m.f].sum(axis=1)) != 3
        return m.f[which_fs]

    def mesh_xsections(self, m, neighborhood=None):
        '''
        Takes a cross section of planar point cloud with a Mesh object.
        Ignore those points which intersect at a vertex - the probability of
        this event is small, and accounting for it complicates the algorithm.

        If 'neighborhood' is provided, use a KDTree to constrain the
        cross section to the closest connected component to 'neighborhood'.

        Params:
            - m:
                Mesh object
            - neigbhorhood:
                M x 3 np.array

        Returns a list of Polylines.
        '''
        from blmath.geometry import Polyline

        # 1: Select those faces that intersect the plane, fs
        fs = self.mesh_intersecting_faces(m)
        if len(fs) == 0:
            return [] # Nothing intersects
        # and edges of those faces
        es = np.vstack((fs[:, (0, 1)], fs[:, (1, 2)], fs[:, (2, 0)]))

        # 2: Find the edges where each of those faces actually cross the plane
        class EdgeMap(object):
            # A quick two level dictionary where the two keys are interchangeable (i.e. a symmetric graph)
            def __init__(self):
                self.d = {} # store indicies into self.values here, to make it easier to get inds or values
                self.values = []
            def _order(self, u, v):
                if u < v:
                    return u, v
                else:
                    return v, u
            def add(self, u, v, val):
                low, high = self._order(u, v)
                if low not in self.d:
                    self.d[low] = {}
                self.values.append(val)
                self.d[low][high] = len(self.values) - 1
            def contains(self, u, v):
                low, high = self._order(u, v)
                if low in self.d and high in self.d[low]:
                    return True
                return False
            def index(self, u, v):
                low, high = self._order(u, v)
                try:
                    return self.d[low][high]
                except KeyError:
                    return None
            def get(self, u, v):
                ii = self.index(u, v)
                if ii is not None:
                    return self.values[ii]
                else:
                    return None

        intersection_map = EdgeMap()

        pts, pt_is_valid = self.line_segment_xsections(m.v[es[:, 0]], m.v[es[:, 1]])
        valid_pts = pts[pt_is_valid]
        valid_es = es[pt_is_valid]
        for val, e in zip(valid_pts, valid_es):
            if not intersection_map.contains(e[0], e[1]):
                intersection_map.add(e[0], e[1], val)
        verts = np.array(intersection_map.values)

        class Graph(object):
            # A little utility class to build a symmetric graph and calculate Euler Paths
            def __init__(self, size):
                self.size = size
                self.d = {}
            def __len__(self):
                return len(self.d)
            def add_edges(self, edges):
                for u, v in edges:
                    self.add_edge(u, v)
            def add_edge(self, u, v):
                assert u >= 0 and u < self.size
                assert v >= 0 and v < self.size
                if u not in self.d:
                    self.d[u] = set()
                if v not in self.d:
                    self.d[v] = set()
                self.d[u].add(v)
                self.d[v].add(u)
            def remove_edge(self, u, v):
                if u in self.d and v in self.d[u]:
                    self.d[u].remove(v)
                if v in self.d and u in self.d[v]:
                    self.d[v].remove(u)
                if v in self.d and len(self.d[v]) == 0:
                    del self.d[v]
                if u in self.d and len(self.d[u]) == 0:
                    del self.d[u]
            def pop_euler_path(self, allow_multiple_connected_components=True):
                # Based on code from Przemek Drochomirecki, Krakow, 5 Nov 2006
                # http://code.activestate.com/recipes/498243-finding-eulerian-path-in-undirected-graph/
                # Under PSF License
                # NB: MUTATES d

                # counting the number of vertices with odd degree
                odd = [x for x in self.d if len(self.d[x])&1]
                odd.append(self.d.keys()[0])
                if not allow_multiple_connected_components and len(odd) > 3:
                    return None
                stack = [odd[0]]
                path = []
                # main algorithm
                while stack:
                    v = stack[-1]
                    if v in self.d:
                        u = self.d[v].pop()
                        stack.append(u)
                        self.remove_edge(u, v)
                    else:
                        path.append(stack.pop())
                return path

        # 4: Build the edge adjacency graph
        G = Graph(verts.shape[0])
        for f in fs:
            # Since we're dealing with a triangle that intersects the plane, exactly two of the edges
            # will intersect (note that the only other sorts of "intersections" are one edge in
            # plane or all three edges in plane, which won't be picked up by mesh_intersecting_faces).
            e0 = intersection_map.index(f[0], f[1])
            e1 = intersection_map.index(f[0], f[2])
            e2 = intersection_map.index(f[1], f[2])
            if e0 is None:
                G.add_edge(e1, e2)
            elif e1 is None:
                G.add_edge(e0, e2)
            else:
                G.add_edge(e0, e1)

        # 5: Find the paths for each component
        components = []
        components_closed = []
        while len(G) > 0:
            path = G.pop_euler_path()
            if path is None:
                raise ValueError("mesh slice has too many odd degree edges; can't find a path along the edge")
            component_verts = verts[path]

            if np.all(component_verts[0] == component_verts[-1]):
                # Because the closed polyline will make that last link:
                component_verts = np.delete(component_verts, 0, axis=0)
                components_closed.append(True)
            else:
                components_closed.append(False)
            components.append(component_verts)

        if neighborhood is None or len(components) == 1:
            return [Polyline(v, closed=closed) for v, closed in zip(components, components_closed)]

        # 6 (optional - only if 'neighborhood' is provided): Use a KDTree to select the component with minimal distance to 'neighborhood'
        from scipy.spatial import cKDTree  # First thought this warning was caused by a pythonpath problem, but it seems more likely that the warning is caused by scipy import hackery. pylint: disable=no-name-in-module

        kdtree = cKDTree(neighborhood)

        # number of components will not be large in practice, so this loop won't hurt
        means = [np.mean(kdtree.query(component)[0]) for component in components]
        return [Polyline(components[np.argmin(means)], closed=True)]


def main():
    import argparse
    from lace.mesh import Mesh

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='filepath to mesh', required=True)
    parser.add_argument('-c', '--cloud', help='display point cloud', required=False, default=False, action='store_true')
    parser.add_argument('-d', '--direction', help='direction of connected component',
                        choices=['N', 'S', 'E', 'W'], default=None, required=False)
    args = parser.parse_args()

    path_to_mesh = args.path
    mesh = Mesh(filename=path_to_mesh, vc='SteelBlue')

    point_on_plane = np.array([0., 1., 0.])

    n1 = np.array([0., 1., 0.])
    p1 = Plane(point_on_plane, n1)

    n2 = np.array([1., 0., 0.])
    p2 = Plane(point_on_plane, n2)

    n3 = np.array([1., 1., 0.])
    n3 /= np.linalg.norm(n3)
    p3 = Plane(point_on_plane, n3)

    n4 = np.array([-1., 1., 0.])
    n4 /= np.linalg.norm(n4)
    p4 = Plane(point_on_plane, n4)

    dirmap = {
        'N': [0., +100., 0.],
        'S': [0., -100., 0.],
        'E': [+100., 0., 0.],
        'W': [-100., 0., 0.],
        None: None,
    }

    neighborhood = dirmap[args.direction]
    if neighborhood != None:
        neighborhood = np.array([neighborhood])

    xs1 = p1.mesh_xsection(mesh, neighborhood=neighborhood)
    xs2 = p2.mesh_xsection(mesh, neighborhood=neighborhood)
    xs3 = p3.mesh_xsection(mesh, neighborhood=neighborhood)
    xs4 = p4.mesh_xsection(mesh, neighborhood=neighborhood)

    lines = [
        polyline.as_lines()
        for polyline in xs1, xs2, xs3, xs4
    ]

    if args.cloud:
        mesh.f = []

    from lace.meshviewer import MeshViewer
    mv = MeshViewer(keepalive=True)
    mv.set_dynamic_meshes([mesh], blocking=True)
    mv.set_dynamic_lines(lines)


if __name__ == '__main__':
    main()
