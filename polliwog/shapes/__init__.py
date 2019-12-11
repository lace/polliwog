"""
Functions for creating sets of triangles to model 3D shapes.

These functions have two possible return types:

- When `ret_unique_vertices_and_faces=True`, they return a vertex array (with
  each vertex listed once) and a face array (i.e. an array of triples of vertex
  indices). This is ideal when using with a mesh library like Lace
  (https://github.com/lace/lace/) or Trimesh (https://trimsh.org/) or when you
  care about the topology.
- When `ret_unique_vertices_and_faces=False`, they return a flattened array
  of triangle coordinates with each vertex repeated. This is useful for
  computation that use flattened triangle coordinates, such as the functions
  in `polliwog.tri`.

See also:
    https://en.wikipedia.org/wiki/Tessellation_(computer_graphics)
"""

from . import _shapes
from ._shapes import *  # noqa: F401,F403

__all__ = _shapes.__all__
