import os
import numpy as np
import pytest
from ._rodrigues import (
    cv2_rodrigues,
    rodrigues_vector_to_rotation_matrix,
    rotation_matrix_to_rodrigues_vector,
)


def load_opencv_examples():
    test_data_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "rodrigues_test_assets.npz"
    )
    with open(test_data_path, "rb") as f:
        data = np.load(f, allow_pickle=True)
        return list(data["test_assets"])


opencv_examples = load_opencv_examples()


@pytest.mark.parametrize("arg,result", opencv_examples)
def test_rodrigues_matches_opencv(arg, result):
    r = cv2_rodrigues(arg)
    np.testing.assert_array_almost_equal(r, result[0])


@pytest.mark.parametrize("arg,result", opencv_examples)
def test_rodrigues_with_derivatives_matches_opencv(arg, result):
    r, dr = cv2_rodrigues(arg, calculate_jacobian=True)
    np.testing.assert_array_almost_equal(r, result[0])
    np.testing.assert_array_almost_equal(dr, result[1])


def test_rodrigues_errors_with_bad_shape():
    with pytest.raises(ValueError):
        cv2_rodrigues(np.array([0.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("arg,result", opencv_examples)
def test_rodrigues_explicit_functions_match_opencv(arg, result):
    if arg.size == 3:
        r = rodrigues_vector_to_rotation_matrix(arg)
    else:
        r = rotation_matrix_to_rodrigues_vector(arg)
    np.testing.assert_array_almost_equal(r, result[0])


@pytest.mark.parametrize("arg,result", opencv_examples)
def test_rodrigues_explicit_functions_with_derivatives_match_opencv(arg, result):
    if arg.size == 3:
        r, dr = rodrigues_vector_to_rotation_matrix(arg, calculate_jacobian=True)
    else:
        r, dr = rotation_matrix_to_rodrigues_vector(arg, calculate_jacobian=True)
    np.testing.assert_array_almost_equal(r, result[0])
    np.testing.assert_array_almost_equal(dr, result[1])


def test_rodrigues_vector_to_rotation_matrix_errors_with_bad_shape():
    with pytest.raises(ValueError):
        rodrigues_vector_to_rotation_matrix(np.array([0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        rodrigues_vector_to_rotation_matrix(np.eye(3))


def test_rotation_matrix_to_rodrigues_vector_errors_with_bad_shape():
    with pytest.raises(ValueError):
        rotation_matrix_to_rodrigues_vector(np.array([0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        rotation_matrix_to_rodrigues_vector(np.array([0.0, 0.0, 0.0]))
