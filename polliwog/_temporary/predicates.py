def isnumeric(x):
    import numpy as np
    if hasattr(x, '__iter__'):
        x = np.asarray(x)
    if isinstance(x, bool) or isinstance(x, np.bool) or (hasattr(x, 'dtype') and x.dtype == np.bool):
        # Special case; bools do support math, but don't match our definition of numeric
        return False
    if not np.all(np.isrealobj(x)):
        # We don't want complex numbers
        return False
    try:
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return False
    except TypeError: # isnan is not supported on some builtin numeric types like Decimal
        pass
    try:
        # See if it supports mathiness
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x+x, x-x, x*x, x/x # pylint: disable=pointless-statement
    except ZeroDivisionError:
        return True
    except (TypeError, Exception): # FIXME pylint: disable=broad-except
        return False
    else:
        return True

def isnumericscalar(x):
    return not hasattr(x, '__iter__') and isnumeric(x)

def isnumericarray(x):
    return hasattr(x, '__iter__') and isnumeric(x)

def is_empty_arraylike(arraylike):
    """Check if arraylike is None or empty.

    Arg:
        arraylike: None or np.ndarray or list or tuple

    Return:
        a boolean indicates if this array is empty
    """
    # pylint: disable=len-as-condition
    import numpy as np
    if arraylike is None:
        return True
    if isinstance(arraylike, np.ndarray) and arraylike.size == 0:
        return True
    if isinstance(arraylike, list) and len(arraylike) == 0:
        return True
    if isinstance(arraylike, tuple) and len(arraylike) == 0:
        return True
    return False
