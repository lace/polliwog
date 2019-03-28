class CoordinateManager(object):
    '''
    Here's the idea:

        coordinate_manager = CoordinateManager()
        coordinate_manager.tag_as('source')
        coordinate_manager.translate(-cube.floor_point)
        coordinate_manager.scale(2)
        coordinate_manager.tag_as('floored_and_scaled')
        coordinate_manager.translate(np.array([0., -4., 0.]))
        coordinate_manager.tag_as('centered_at_origin')

        coordinate_manager.source = cube
        centered_mesh = coordinate_manager.centered_at_origin

    '''

    def __init__(self):
        from blmath.geometry.transform.composite import CompositeTransform
        self.__dict__.update({
            # A map from tag names to indices into the transform stack.
            'tags_to_indices': {},

            # Our currently set points, and the tag at which they belong.
            'points_tag': None,
            'points': None,

            # Our worthy collaborator.
            '_transform': CompositeTransform(),
        })

    def append_transform4(self, *args, **kwargs):
        self._transform.append_transform4(*args, **kwargs)
    def append_transform3(self, *args, **kwargs):
        self._transform.append_transform3(*args, **kwargs)
    def scale(self, *args, **kwargs):
        self._transform.scale(*args, **kwargs)
    def convert_units(self, *args, **kwargs):
        self._transform.convert_units(*args, **kwargs)
    def translate(self, *args, **kwargs):
        self._transform.translate(*args, **kwargs)
    def reorient(self, *args, **kwargs):
        self._transform.reorient(*args, **kwargs)
    def rotate(self, *args, **kwargs):
        self._transform.rotate(*args, **kwargs)

    def tag_as(self, name):
        '''
        Give a name to the current state.

        '''
        # The indices of CompositeTransform are 0-based, which means the first
        # transform is transform 0.
        #
        # In CoordinateManager, we refer to the initial state -- i.e. no
        # transforms -- with 0. The state after the first transform is 1.
        # After two transforms, 2. Put another way: we refer to a given state
        # with a value that is *one more than* the index of the state's last
        # transform. This is a little strange, but we need the extra zero
        # state. And it ends up playing nicely with array slicing.
        self.tags_to_indices[name] = len(self._transform.transforms)

    def do_transform(self, points_or_mesh, from_tag, to_tag):
        from copy import copy

        if hasattr(points_or_mesh, 'v'):
            points = points_or_mesh.v
            # Can't run the transform if there are no vertices.
            if points is None:
                return points_or_mesh
        else:
            points = points_or_mesh

        try:
            from_index = self.tags_to_indices[from_tag]
            to_index = self.tags_to_indices[to_tag]
        except KeyError as e:
            raise KeyError("No such tag: %s" % e.message)

        if from_index == to_index:
            result_points = points
        elif from_index < to_index:
            from_range = from_index, to_index
            result_points = self._transform(points, from_range=from_range)
        else:
            from_range = to_index, from_index
            result_points = self._transform(points, from_range=from_range, reverse=True)

        if hasattr(points_or_mesh, 'v'):
            # for lace or those object with copy method, invoke its own copy method
            # otherwise just shallow copy
            result_mesh = points_or_mesh.copy() if hasattr(points_or_mesh, 'copy') else copy(points_or_mesh)
            result_mesh.v = result_points

            return result_mesh
        else:
            return result_points

    def __setattr__(self, name, value):
        '''
        value: Either an nx3 array of points or an instance of Mesh.

        '''
        if name not in self.tags_to_indices:
            raise AttributeError("No such tag: %s" % name)

        self.__dict__['points_tag'] = name
        self.__dict__['points_or_mesh'] = value

    def __getattr__(self, name):
        if self.points_tag is None:
            raise ValueError('Must set a value before trying to read one')

        return self.do_transform(
            points_or_mesh=self.points_or_mesh,
            from_tag=self.points_tag,
            to_tag=name
        )
