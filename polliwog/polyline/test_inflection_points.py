from collections import namedtuple
import numpy as np
import pytest
import vg
from ._inflection_points import inflection_points, point_of_max_acceleration


def generate_samples(fn, domain):
    xmin, xmax = domain
    xs = np.arange(xmin, xmax, 0.1).reshape(-1, 1)
    ys = fn(xs)
    zs = np.zeros_like(xs)

    return np.hstack([xs, ys, zs])


InflectionPointExample = namedtuple(
    "InflectionPointExample",
    ["fn", "domain", "inflection_points", "point_of_max_acceleration"],
)

examples = [
    InflectionPointExample(
        fn=lambda x: (x ** 6) / 30 - (x ** 5) / 20 - (x ** 4) + 3 * x + 20,
        domain=(-10, 10),
        inflection_points=np.array([-3.0, 4.0]),
        point_of_max_acceleration=9.8,
    ),
    InflectionPointExample(
        fn=lambda x: x ** 4,
        domain=(-3, 3),
        inflection_points=np.array([]),
        point_of_max_acceleration=2.8,
    ),
    InflectionPointExample(
        fn=lambda x: x ** 3,
        domain=(-3, 3),
        inflection_points=np.array([0.0]),
        point_of_max_acceleration=2.8,
    ),
    InflectionPointExample(
        fn=lambda x: x ** 3 - 3 * x ** 2 - 144 * x + 432,
        domain=(-12, 12),
        inflection_points=np.array([1.0]),
        point_of_max_acceleration=11.8,
    ),
    InflectionPointExample(
        fn=lambda x: np.sin(2 * x),
        domain=(0, 2 * np.pi),
        inflection_points=np.array([1.5, 3.1, 4.7]),
        point_of_max_acceleration=5.5,
    ),
    InflectionPointExample(
        fn=lambda x: -(x ** 2),
        domain=(0, 5),
        inflection_points=np.array([]),
        point_of_max_acceleration=None,
    ),
]


def test_inflection_points():
    for example in examples:
        samples = generate_samples(fn=example.fn, domain=example.domain)
        result = inflection_points(
            points=samples, rise_axis=vg.basis.y, run_axis=vg.basis.x
        )
        np.testing.assert_array_almost_equal(
            result[:, 0], example.inflection_points, decimal=1
        )
        np.testing.assert_array_almost_equal(result[:, 1], example.fn(result[:, 0]))
        np.testing.assert_array_almost_equal(
            result[:, 2], np.zeros_like(example.inflection_points)
        )


def test_point_of_max_acceleration():
    for example in examples:
        samples = generate_samples(fn=example.fn, domain=example.domain)
        result = point_of_max_acceleration(
            points=samples,
            rise_axis=vg.basis.y,
            run_axis=vg.basis.x,
            subdivide_by_length=0.1,
        )
        if example.point_of_max_acceleration is None:
            assert result is None
        else:
            np.testing.assert_array_almost_equal(
                result[0], example.point_of_max_acceleration, decimal=1
            )
            np.testing.assert_array_almost_equal(
                result[1], example.fn(result[0]), decimal=1
            )
            np.testing.assert_array_almost_equal(result[2], np.array([0]))

    with pytest.raises(ValueError, match="At least two points are required"):
        point_of_max_acceleration(
            points=np.array([vg.basis.x]), rise_axis=vg.basis.y, run_axis=vg.basis.x,
        )
