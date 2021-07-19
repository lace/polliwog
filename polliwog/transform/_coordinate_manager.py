from vg.compat import v2 as vg
from ._composite_transform import CompositeTransform


class CoordinateManager(object):
    """
    Example:
        >>> coordinate_manager = CoordinateManager()
        >>> coordinate_manager.tag_as('source')
        >>> coordinate_manager.translate(-cube.floor_point)
        >>> coordinate_manager.uniform_scale(2)
        >>> coordinate_manager.tag_as('floored_and_scaled')
        >>> coordinate_manager.translate(np.array([0., -4., 0.]))
        >>> coordinate_manager.tag_as('centered_at_origin')

        >>> coordinate_manager.source = cube
        >>> centered_mesh = coordinate_manager.centered_at_origin
    """

    def __init__(self):
        self.__dict__.update(
            {
                # A map from tag names to indices into the transform stack.
                "_tags_to_indices": {},
                # Our currently set points, and the tag at which they belong.
                "_points_tag": None,
                "_points": None,
                # Our worthy collaborator.
                "_transform": CompositeTransform(),
            }
        )

    def append_transform(self, *args, **kwargs):
        self._transform.append_transform(*args, **kwargs)

    def uniform_scale(self, *args, **kwargs):
        self._transform.uniform_scale(*args, **kwargs)

    def non_uniform_scale(self, *args, **kwargs):
        self._transform.non_uniform_scale(*args, **kwargs)

    def convert_units(self, *args, **kwargs):
        self._transform.convert_units(*args, **kwargs)

    def flip(self, *args, **kwargs):
        self._transform.flip(*args, **kwargs)

    def translate(self, *args, **kwargs):
        self._transform.translate(*args, **kwargs)

    def reorient(self, *args, **kwargs):
        self._transform.reorient(*args, **kwargs)

    def rotate(self, *args, **kwargs):
        self._transform.rotate(*args, **kwargs)

    def tag_as(self, name):
        """
        Give a name to the current state.

        """
        # The indices of CompositeTransform are 0-based, which means the first
        # transform is transform 0.
        #
        # In CoordinateManager, we refer to the initial state -- i.e. no
        # transforms -- with 0. The state after the first transform is 1.
        # After two transforms, 2. Put another way: we refer to a given state
        # with a value that is *one more than* the index of the state's last
        # transform. This is a little strange, but we need the extra zero
        # state. And it ends up playing nicely with array slicing.
        self._tags_to_indices[name] = len(self._transform.transforms)

    def do_transform(self, points, from_tag, to_tag):
        try:
            from_index = self._tags_to_indices[from_tag]
            to_index = self._tags_to_indices[to_tag]
        except KeyError as e:
            tag = e.args[0]
            raise KeyError("No such tag: {}".format(tag))

        if from_index == to_index:
            return points
        elif from_index < to_index:
            from_range = from_index, to_index
            return self._transform(points, from_range=from_range)
        else:
            from_range = to_index, from_index
            return self._transform(points, from_range=from_range, reverse=True)

    def __setattr__(self, name, points):
        """
        value: An nx3 array of points or an instance of Mesh.
        """
        if name not in self._tags_to_indices:
            raise AttributeError("No such tag: %s" % name)

        vg.shape.check(locals(), "points", (-1, 3))

        self.__dict__["_points_tag"] = name
        self.__dict__["_points"] = points

    def __getattr__(self, name):
        from_tag = self._points_tag
        if from_tag is None:
            raise ValueError("Must set the points before trying to read them")

        return self.do_transform(
            points=self._points, from_tag=self._points_tag, to_tag=name
        )
