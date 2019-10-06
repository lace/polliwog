import vg


def columnize(arr, shape=(-1, 3), name=None):
    """
    Helper for functions which may accept many stacks of three points (kx3)
    returning a stack of results, or a single set of three points (3x1)
    returning a single result.

    Returns the points as kx3, and a `transform_result` function which can
    be applied to the result. It picks off the first result in the 3x1 case.

    Not limited to kx3; this can be used different dimensional shapes like
    kx4, or higher dimensional shapes like kx3x3.
    """
    if not isinstance(shape, tuple):
        raise ValueError("shape should be a tuple")
    name = name or "arr"

    if arr.ndim == len(shape):
        vg.shape.check_value(arr, shape, name=name)
        return arr, True, lambda x: x
    else:
        vg.shape.check_value(arr, shape[1:], name=name)
        return arr.reshape(*shape), False, lambda x: x[0]


# TODO: After dropping Python 2, make `name=None` a regular kwarg.
def check_shape_any(arr, *shapes, **kwargs):
    if len(shapes) == 0:
        raise ValueError("At least one shape is required")
    name = kwargs.get("name", "arr")
    for shape in shapes:
        try:
            vg.shape.check_value(arr, shape, name=name)
            return
        except ValueError:
            pass

    if name is None:
        preamble = "Expected an array"
    else:
        preamble = "{} must be an array".format(name)

    shape_choices = ", ".join(
        shapes[:-2] + (" or ".join([str(shapes[-2]), str(shapes[-1])]),)
    )

    if arr is None:
        raise ValueError("{} with shape {}; got None".format(preamble, shape_choices))
    else:
        raise ValueError(
            "{} with shape {}; got {}".format(preamble, shape_choices, arr.shape)
        )
