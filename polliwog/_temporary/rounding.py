def round_to(value, nearest):
    '''
    Round a number to the nearest specified increment.

    e.g.:

        >>> round_to(3.8721, 0.05)
        3.85
        >>> round_to(3.8721, 0.1)
        3.9
        >>> round_to(3.8725, 0.25)
        3.75
        >>> round_to(3.8725, 2.0)
        4

    Use reciprocal due to floating point weirdness:

        >>> value, nearest = 3.9, 0.1
        >>> round(value / nearest) * nearest
        3.9000000000000004
        >>> round(value / nearest) / (1.0 / nearest)
        3.9

    '''
    reciprocal = 1.0 / nearest
    return round(value / nearest) / reciprocal

def rounded_list(iterable, precision=0):
    return [(None if x is None else round(x, precision)) for x in iterable]
