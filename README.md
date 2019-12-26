# polliwog

[![version](https://img.shields.io/pypi/v/polliwog.svg?style=flat-square)][pypi]
[![python versions](https://img.shields.io/pypi/pyversions/polliwog.svg?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/polliwog.svg?style=flat-square)][pypi]
[![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat-square)][coverage]
[![build](https://img.shields.io/circleci/project/github/lace/polliwog/master.svg?style=flat-square)][build]
[![docs build](https://img.shields.io/readthedocs/polliwog.svg?style=flat-square)][docs build]
[![code style](https://img.shields.io/badge/code%20style-black-black.svg?style=flat-square)][black]

2D and 3D computational geometry library which scales from prototyping to production.

Includes vectorized geometric operations, transforms, and primitives like planes,
polygonal chains, and axis-aligned bounding boxes. Implemented in pure Python/NumPy.
Lightweight and fast.

The goals of this project are:

- Keep dependencies light and deployment flexible.
- Keep the library working in current versions of Python and other tools.
- Respond to community contributoions.
- Eventually provide a complete set of functionality for this problem domain.

[pypi]: https://pypi.org/project/polliwog/
[coverage]: https://github.com/lace/polliwog/blob/master/.coveragerc#L2
[build]: https://circleci.com/gh/lace/polliwog/tree/master
[docs build]: https://polliwog.readthedocs.io/en/latest/
[black]: https://black.readthedocs.io/en/stable/

## Features

Geometric operations, transforms, and primitives, in 2D and 3D.

The [most commonly used of these](__init__.py) are directly imported into
`polliwog`.

- [polliwog.Box](polliwog/box/box.py) represents an axis-aligned
  cuboid.
- [polliwog.Plane](polliwog/plane/plane.py) represents a 2-D plane in
  3-space (not a hyperplane).
- [polliwog.Polyline](polliwog/polyline/polyline.py) represents an
  unconstrained polygonal chain in 3-space.

`polliwog.transform` includes code for 3D transforms.

- [polliwog.transform.CompositeTransform](polliwog/transform/composite.py)
  represents a composite transform using homogeneous coordinates. (Thanks avd!)
- [polliwog.transform.CoordinateManager](polliwog/transform/coordinate_manager.py)
  provides a convenient interface for named reference frames within a stack of
  transforms and projecting points from one reference frame to another.
- [polliwog.transform.find_rigid_transform](polliwog/transform/rigid_transform.py)
  finds a rotation and translation that closely transforms one set of points to
  another. Its cousin `find_rigid_rotation` does the same, but only allows
  rotation, not translation.
- [polliwog.transform.rotation.rotation_from_up_and_look](polliwog/transform/rotation.py)
  produces a rotation matrix that gets a mesh into the canonical reference frame
  from "up" and "look" vectors.

Other modules:

- [polliwog.tri.barycentric](polliwog/tri/barycentric.py) provides a function for
  projecting a point to a triangle using barycentric coordinates.
- [polliwog.segment](polliwog/segment/segment.py) provides functions for working with
  line segments in n-space.


## Installation

```sh
pip install polliwog
```

## Usage

```py
import numpy as np
from polliwog import Polyline

# ...
```


## Contribute

- Issue Tracker: https://github.com/lace/polliwog/issues
- Source Code: https://github.com/lace/polliwog

Pull requests welcome!


## Support

If you are having issues, please let us know.


## Acknowledgements

This collection was developed at Body Labs and includes a combination of code
developed at Body Labs, from legacy code and significant new portions by
[Eric Rachlin][], [Alex Weiss][], and [Paul Melnikow][]. It was extracted
from the Body Labs codebase and open-sourced by [Alex Weiss][] into a library
called [blmath][], which was subsequently [forked by Paul Melnikow][blmath fork].
This library and the 3D geometry and linear-algebra toolbelt [vg][] were later
extracted.

[eric rachlin]: https://github.com/eerac
[alex weiss]: https://github.com/algrs
[paul melnikow]: https://github.com/paulmelnikow
[blmath]: https://github.com/bodylabs/blmath
[blmath fork]: https://github.com/metabolize/blmath
[vg]: https://github.com/lace/vg


## License

The project is licensed under the two-clause BSD license.
