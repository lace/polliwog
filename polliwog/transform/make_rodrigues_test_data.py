#!/usr/bin/env python

import os
import sys
import click
import numpy as np

try:
    import cv2
except ImportError:
    print(
        "Error: This script requires opencv, "
        "which is not a runtime or dev dependency "
        "and needs to be installed separately."
    )
    exit(-1)
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from polliwog.transform.rotation import (
    euler,
)  # noqa: E402 Needs to happen after the modification of sys.path


THINGS_TO_TEST = [
    np.eye(3),
    np.array([0.0, 0.0, 0.0]),
    np.array([[0.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 0.0]]).T,
    np.array([0.5, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([10.0, 11.0, 12.0]),  # make sure we work for rotations > 2pi
    euler([45, 0, 0]),  # these are rotation matrices
    euler([0, 32, 0]),
    euler([0, 0, 67]),
    euler([45, 37, 19]),
    np.array(
        [0.644164, 0.726324, 0.051697]
    ),  # approx the rodrigues of the previous matrix
    # test special cases:
    np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]),
    np.array([[0.0, -1.0, 2.0], [-1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]),
    np.array([[0.0, 1.0, -2.0], [1.0, 0.0, 3.0], [-2.0, 3.0, 0.0]]),
    np.array([[0.1, 1.0, 2.0], [1.0, 1.7, -3.0], [2.0, -3.0, 2.8]]),
]


@click.command()
def main():
    print("Building Rodrigues test assets")
    print(f"OpenCV Version: {cv2.__version__}")
    test_assets = [(x, cv2.Rodrigues(x)) for x in THINGS_TO_TEST]
    print(f"Produced {len(test_assets)} test assets")
    test_data_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "rodrigues_test_assets.npz"
    )
    print(f"Writing to {test_data_path}")
    with open(test_data_path, "wb") as f:
        np.savez(f, version=cv2.__version__, test_assets=test_assets)
    print(f"Checking {test_data_path}")
    with open(test_data_path, "rb") as f:
        data = np.load(f, allow_pickle=True)
        print(f"Found opencv version {data['version']}")
        print(f"Found {len(data['test_assets'])} assets")


if __name__ == "__main__":
    main()
