import numpy as np


def find_repeats(arr, wrap=False):
    """
    Find the indices of the elements in an array which are equal to the
    elements which immediately precede them.

    To make it easy to work with open and closed polygonal chains, the shape
    of the output will always have the same length as the shape of the input.

    When `wrap` is `True` the first value will be compared to the last value.
    When `wrap` is `False` the first index of the result will always be False.
    """
    if wrap:
        return np.roll(arr, 1) == arr
    else:
        return np.concatenate(([False], arr[:-1] == arr[1:]))


def find_changes(arr, wrap=False):
    """
    Find the indices of the elements in an array which differ from the
    elements which immediately precede them.

    To make it easy to work with open and closed polygonal chains, the shape
    of the output will always have the same length as the shape of the input.

    When `wrap` is `True` the first value will be compared to the last value.
    When `wrap` is `False` the first index of the result will always be False.
    """
    if wrap:
        return np.roll(arr, 1) != arr
    else:
        return np.concatenate(([False], arr[:-1] != arr[1:]))
