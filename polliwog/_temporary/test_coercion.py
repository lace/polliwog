import unittest
import numpy as np


class TestCoercion(unittest.TestCase):
    def test_as_numeric_array_three_by_one(self):
        from .coercion import as_numeric_array

        good = [np.array([1, 2, 3])]

        good_iff_allow_none = [None]

        good_iff_allow_none_and_empty_as_none = [tuple(), []]

        bad = [
            (1,),
            (1, 2),
            (1, 2, 3, 4),
            [1],
            [1, 2],
            [1, 2, 3, 4],
            np.array([1, 2]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3]).reshape(3, 1),
            ("foo", "bar", "baz"),
            ["foo", "bar", "baz"],
        ]

        for v in good:
            as_numeric_array(v, shape=(3,), allow_none=True)
            as_numeric_array(v, shape=(3,), allow_none=False)

        for v in good_iff_allow_none:
            as_numeric_array(v, shape=(3,), allow_none=True)
            with self.assertRaises(ValueError):
                as_numeric_array(v, shape=(3,), allow_none=False)

        for v in good_iff_allow_none_and_empty_as_none:
            as_numeric_array(v, shape=(3,), allow_none=True, empty_as_none=True)
            with self.assertRaises(ValueError):
                as_numeric_array(v, shape=(3,), allow_none=False, empty_as_none=True)

        for v in bad:
            with self.assertRaises(ValueError):
                as_numeric_array(v, shape=(3,), allow_none=True)
            with self.assertRaises(ValueError):
                as_numeric_array(v, shape=(3,), allow_none=False)
