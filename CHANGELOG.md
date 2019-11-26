# Changelog

## 0.12.0 (Nov. 25, 2019)

### BREAKING CHANGES

- Require Python 3.
- Polyline:
    - Rename `flip()` to `flipped()`.
    - Rename `oriented_along()` to `aligned_with()` and drop `reverse`
      parameter.
    - Rename `bisect_edges()` to `with_segments_bisected()`.
    - Rename `cut_by_plane()` to `sliced_by_plane()`.
    - Rename `reindexed()` to `rolled()`.
- CompositeTransform:
    - Require `np.array` inputs, not lists.
    - Rename some arguments.
    - Remove special support for `lace.mesh.Mesh`. This functionality can be
      restored as a convenience function on Mesh itself.
- `rotation_from_up_and_look()`: Require `np.array` input, not list.
- Consolidate `tri.barycentric`, `tri.contains`, and `tri.surface_normals`
  into `tri.functions`.
- Rename `tri.arity` to `tri.quad_faces`.
- Remove `transform.translation()`.
- Remove `transform.composite.convert_44_to_33()`. Make `convert_33_to_44()`
  private for now.

## New features

- Polyline:
    - Add `index_of_vertex()`.
    - Add `with_insertions()`.
    - Add `sliced_at_points()`.
    - Add `sliced_at_indices()`.
    - `join()`: Add `is_closed` parameter.
- Add `transform.apply_affine_transform()`.

## Other maintenance

- Auto-generate documentation and start to improve them. They aren't 100% but
  they're a good part of the way there:
  https://polliwog.readthedocs.io/en/latest/
- Consolidate duplicate implementations
  `plane.functions.plane_normal_from_points` and
  `tri.functions.surface_normals`.
- Replace pyflakes with flake8.
- Refactor some array shape validation code.
- Remove `setter_property` decorator.
- Remove `rotate_to_xz_plane()`.
- Stop using `vg.matrix` which is being deprecated.
- Replace pint with [ounce][].

[ounce]: https://github.com/lace/ounce


## 0.11.0 (Oct. 27, 2019)

### BREAKING CHANGES

- Polyline: Rename `closed` property to `is_closed`.
- Box: Rename `shape` to `size`.
- Rename `line_intersect()` to `line_intersect2()` and return None instead of
  nan's.

## New features

- Polyline: Add `oriented_along()` method.
- Polyline: Add `bounding_box` property.
- Polyline: Add `join()` class method.

## Bug fixes

- Fix `check_shape_any()` error messages and add tests.

## Other maintenance

- Require `vg` 1.5+.
- Omit tests from PyPI distribution.

## 0.10.0 (Oct. 12, 2019)

- `Polyline.cut_by_plane()`: Handle vertices which lie in the plane.

## 0.9.0 (Oct. 6, 2019)

- Add `segment.segment.closest_point_of_line_segment()`
- Add `line.functions.project_to_line()`

## 0.8.0 (Oct. 5, 2019)

- Polyline:
    - Add `num_v` and `num_e` properties.
    - Remove `as_lines()` and references to lace.
    - Refactor `segment_lengths` using `vg.euclidean_distance()`.
    - Vectorize `partition_by_length()`.
- Fix `inflection_points` and add test coverage.
- Add `tri.arity.quads_to_tris()` (from lace).
- Add vectorized plane functions.
- Improve test coverage.
- Don't ship the tests.

## 0.7.0 (June 30, 2019)

- Polyline: Add `bisect_edges()` method.
- Return `np.ndarray`s from shapes functions.

## 0.6.0 (May 29, 2019)

- Add Line primitive.
- Add `polliwog.tri.contains.planar_point`.
- Update vg dependency (now requires 1.2.0).

## 0.5.0 (Apr 4, 2019)

- Update vg dependency (now requires 1.0.0)

## 0.4.0 (Apr 3, 2019)

- Replace most np array coercion with shape checking.
- Use pint for unit conversion.
- Remove most modules from \_temporary.
- Fix erroneous in rigid_transform.

## 0.3.0 (Apr 3, 2019)

- Rewrite `polliwog.tri.barycentric` with new function signature.
- Improve test coverage.

## 0.2.1 (Mar 31, 2019)

- Fix `pip install polliwog`.

## 0.2.0 (Mar 28, 2019)

Initial release.
