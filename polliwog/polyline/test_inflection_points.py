import numpy as np
import vg
from .inflection_points import inflection_points

# Inflection points at x = -3 and x = 4.
poly_fn = lambda x: (x ** 6) / 30 - (x ** 5) / 20 - (x ** 4) + 3 * x + 20


def generate_samples():
    xs = np.arange(-10, 10, 0.1).reshape(-1, 1)
    ys = poly_fn(xs)
    zs = np.zeros_like(xs)

    return np.hstack([xs, ys, zs])


def test_inflection_points():
    samples = generate_samples()
    result = inflection_points(points=samples, axis=vg.basis.y, span=vg.basis.x)
    expected_xs = np.array([-3, 4])
    np.testing.assert_array_almost_equal(result[:, 0], expected_xs, decimal=1)
    np.testing.assert_array_almost_equal(result[:, 1], poly_fn(result[:, 0]))
    np.testing.assert_array_almost_equal(result[:, 2], np.zeros_like(expected_xs))
