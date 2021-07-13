# polliwog

[![version](https://img.shields.io/pypi/v/polliwog?style=flat-square)][pypi]
[![python versions](https://img.shields.io/pypi/pyversions/polliwog?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/polliwog?style=flat-square)][pypi]
[![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat-square)][coverage]
[![build](https://img.shields.io/circleci/project/github/lace/polliwog/main?style=flat-square)][build]
[![docs build](https://img.shields.io/readthedocs/polliwog?style=flat-square)][docs build]
[![code style](https://img.shields.io/badge/code%20style-black-black?style=flat-square)][black]

2D and 3D computational geometry library.

Includes vectorized geometric operations, transforms, and primitives like
planes, polygonal chains, and axis-aligned bounding boxes. Implemented in pure
Python/NumPy. Lightweight and fast.

See the complete API reference: https://polliwog.dev/

Like its lower-level counterpart, the vector-geometry and linear-algebra
toolbelt `vg`, this project is designed to scale from prototyping to production.

The goals of this project are:

- Provide a complete set of functionality for this problem domain, with full
  documentation.
- Provide 100% test coverage of all code paths and use cases.
- Keep dependencies light and deployment flexible.
- Keep the library working with current versions of Python and other tools.
- Respond to community contributions.

[pypi]: https://pypi.org/project/polliwog/
[coverage]: https://github.com/lace/polliwog/blob/main/.coveragerc#L2
[build]: https://circleci.com/gh/lace/polliwog/tree/main
[docs build]: https://polliwog.readthedocs.io/en/latest/
[black]: https://black.readthedocs.io/en/stable/


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


## Development

After cloning the repo, run `./bootstrap.zsh` to initialize a virtual
environment with the project's dependencies.

Subsequently, run `./dev.py install` to update the dependencies.


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
