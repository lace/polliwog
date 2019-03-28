def as_numeric_array(a, shape=None, allow_none=False, empty_as_none=False, dtype=None):
    """
    Coerce a array-like input to a numpy ndarray, and validate it.

    a: An array-like input.

    shape: Provide a shape tuple here to validate the shape of the
      array. To require 3 by 1, pass (3,). To require n by 3, pass
      (-1, 3).

    allow_none: When True, and a is None, return None. When False
      and a is None, raise ValueError.

    empty_as_none: For legacy use in Mesh. When a is an empty array,
      print a warning and return None.

    dtype: Coerces to a specific numpy data type, such as np.float64
      or np.uint64. See np.dtype.

    """
    import numpy as np
    from .predicates import isnumericarray

    if a is None:
        if allow_none:
            return None
        else:
            raise ValueError("Value is None")
    a = np.asarray(a, dtype=dtype)
    if not isnumericarray(a):
        raise ValueError("Must be numeric")
    if allow_none and empty_as_none:
        if a.shape == (0,):
            import warnings

            warnings.warn(
                "To clear this value, set it to None instead of []",
                DeprecationWarning,
                stacklevel=2,
            )
            return as_numeric_array(None, shape, allow_none, dtype)
    if shape is not None:
        if len(shape) != len(a.shape):
            raise ValueError("Shape mismatch: expected %s, got %s" % (shape, a.shape))
        if all([exp == -1 or exp == dim for exp, dim in zip(shape, a.shape)]):
            pass
        elif all([exp == -1 or exp == dim for exp, dim in zip(shape, a.T.shape)]):
            a = a.T
        else:
            raise ValueError(
                "Dimension mismatch: expected %s, got %s" % (shape, a.shape)
            )
    return a
