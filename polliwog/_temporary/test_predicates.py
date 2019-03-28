import unittest
import numpy as np
from .predicates import is_empty_arraylike


class TestPredicates(unittest.TestCase):
    def test_predicates(self):
        from .predicates import isnumeric, isnumericarray

        self.assertTrue(isnumeric(42))
        self.assertTrue(isnumeric(42.0))
        self.assertFalse(isnumeric("42"))

        self.assertTrue(isnumericarray(np.array([1, 2, 3])))
        self.assertTrue(isnumericarray([1, 2, 3]))
        self.assertTrue(isnumericarray([1, 2.0, 3]))
        self.assertFalse(isnumericarray([1, 2.0, "3"]))
        self.assertFalse(isnumericarray(42))

    def test_predicates_some_more(self):
        import decimal
        from .predicates import isnumeric, isnumericarray, isnumericscalar

        def assertNumericArray(val):
            self.assertTrue(isnumeric(val))
            self.assertTrue(isnumericarray(val))
            self.assertFalse(isnumericscalar(val))

        def assertNumericScalar(val):
            self.assertTrue(isnumeric(val))
            self.assertFalse(isnumericarray(val))
            self.assertTrue(isnumericscalar(val))

        def assertNonNumeric(val):
            self.assertFalse(isnumeric(val))
            self.assertFalse(isnumericarray(val))
            self.assertFalse(isnumericscalar(val))

        assertNumericScalar(5)
        assertNumericScalar(123.456)
        assertNumericScalar(decimal.Decimal("123.456"))

        assertNumericArray([1, 2, 3])
        assertNumericArray(np.array([1, 2, 3]))
        a = np.array([1, 2.3, 4.5, 6.7, 8.9])
        assertNumericArray(a)
        assertNumericScalar(a[1])
        assertNumericArray(np.array([np.array([1]), np.array([2])]))

        assertNonNumeric(True)
        assertNonNumeric("123.345")
        assertNonNumeric(np.nan)
        assertNonNumeric(np.inf)
        assertNonNumeric(np.array(["1"]))
        assertNonNumeric(np.array(["foo", "bar", "baz"]))
        assertNonNumeric([1, "2", 3])
        assertNonNumeric(1 + 1j)
        assertNonNumeric(np.array([1 + 1j]))
        assertNonNumeric(1 + 0j)
        assertNonNumeric(np.array([1 + 0j]))
        assertNonNumeric([True, False, True])
        assertNonNumeric(np.array([True, False, True]))

    def test_is_empty_arraylike_none(self):
        array = None
        self.assertTrue(is_empty_arraylike(array))

    def test_is_empty_arraylike_list(self):
        array = []
        self.assertTrue(is_empty_arraylike(array))

    def test_is_empty_arraylike_non_empty_list(self):
        array = [1]
        self.assertFalse(is_empty_arraylike(array))

    def test_is_empty_arraylike_tuple(self):
        array = ()
        self.assertTrue(is_empty_arraylike(array))

    def test_is_empty_arraylike_non_empty_tuple(self):
        array = (1,)
        self.assertFalse(is_empty_arraylike(array))

    def test_is_empty_arraylike_ndarray(self):
        array = np.array([])
        self.assertTrue(is_empty_arraylike(array))

    def test_is_empty_arraylike_non_empty_ndarray(self):
        array = np.array([1])
        self.assertFalse(is_empty_arraylike(array))
