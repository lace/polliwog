import unittest


class TestRounding(unittest.TestCase):
    def test_round_to(self):
        from .rounding import round_to

        self.assertEqual(round_to(3.8721, 0.05), 3.85)
        self.assertEqual(round_to(3.8721, 0.1), 3.9)
        self.assertEqual(round_to(3.8721, 0.25), 3.75)
        self.assertEqual(round_to(3.8721, 2.0), 4)
        self.assertEqual(round_to(3.8721, 2), 4)
        self.assertEqual(round_to(4, 0.33), 3.96)
