import numpy as np
import vg
from .surface_normals import surface_normal


def test_surface_normal_single():
    points = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

    np.testing.assert_allclose(
        surface_normal(points), np.array([3 ** -0.5, 3 ** -0.5, 3 ** -0.5])
    )

    np.testing.assert_allclose(
        surface_normal(points, normalize=False), np.array([9.0, 9.0, 9.0])
    )


def test_surface_normal_vectorized():
    from .shapes import create_triangular_prism

    p1 = np.array([3.0, 0.0, 0.0])
    p2 = np.array([0.0, 3.0, 0.0])
    p3 = np.array([0.0, 0.0, 3.0])
    vertices = create_triangular_prism(p1, p2, p3, 1.0)

    expected_normals = vg.normalize(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -2.0],
                [1.0, 1.0, -2.0],
                [-2.0, 1.0, 1.0],
                [-2.0, 1.0, 1.0],
                [1.0, -2.0, 1.0],
                [1.0, -2.0, 1.0],
                [-1.0, -1.0, -1.0],
            ]
        )
    )

    np.testing.assert_allclose(surface_normal(vertices), expected_normals)
