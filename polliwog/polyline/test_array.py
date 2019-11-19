import numpy as np
from ._array import find_changes, find_repeats


def test_find_repeats():
    example = np.array([0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(
        find_repeats(example, wrap=False),
        np.array(
            [
                False,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                True,
            ]
        ),
    )
    np.testing.assert_array_equal(
        find_repeats(example, wrap=True),
        np.array(
            [
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                True,
            ]
        ),
    )

    example[-1] = 1
    np.testing.assert_array_equal(
        find_repeats(example, wrap=False),
        np.array(
            [
                False,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
            ]
        ),
    )
    np.testing.assert_array_equal(
        find_repeats(example, wrap=True),
        np.array(
            [
                False,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
            ]
        ),
    )


def test_find_changes():
    example = np.array([0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(
        find_changes(example, wrap=False),
        np.array(
            [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                False,
            ]
        ),
    )
    np.testing.assert_array_equal(
        find_changes(example, wrap=True),
        np.array(
            [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                False,
            ]
        ),
    )

    example[-1] = 1
    np.testing.assert_array_equal(
        find_changes(example, wrap=False),
        np.array(
            [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
            ]
        ),
    )
    np.testing.assert_array_equal(
        find_changes(example, wrap=True),
        np.array(
            [
                True,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
            ]
        ),
    )
