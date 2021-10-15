# Changelog

## 2.1.0 (Oct. 15, 2021)

## New features

- `polliwog.plane.slice_triangles_by_plane()`: Optionally return mapping of new
  faces to old.


## 2.0.0 (Oct. 4, 2021)

## BREAKING CHANGES

- Functions which accept triangle face indices require the dtype to be
  `np.int64`.

While this restriction may be a little inconvenient for the caller, it improves
interoperability and performance, simplifies the implementation, and produces
more predictable return values. It's recommended that consuming applications
store all face indices using this dtype.


## 1.2.0 (Oct. 4, 2021)

## New features

- `polliwog.plane.slice_triangles_by_plane()`: Add support for slicing a
  submesh.


## 1.1.0 (Sep. 23, 2021)

## New features

- Add `polliwog.plane.slice_triangles_by_plane()` function, with implementation
  adapted from Trimesh.


## 1.0.1 (Aug. 27, 2021)

## Bug fixes

- `Polyline.point_along_path()`: Fix issue where non-unit length segments
  compute the wrong point. (For segments longer than 1, the point was sometimes
  entirely off the polyline.)


## 1.0.0 (Jul. 18, 2021)

- Upgrade `vg` dependency to `>=2.0`.


## 1.0.0b14 (Jun. 9, 2021)

## BREAKING CHANGES

- Remove `polliwog.__version__`.

## Other changes

- Upgrade `vg` dependency to `>=1.11.1`.


## 1.0.0b13 (Jun. 9, 2021)

## New features

- Polyline: Vectorize `point_along_path()` and support closed polylines.


## 1.0.0b12 (Apr. 15, 2021)

## New features

- Add `polliwog.tri.edges_of_faces` function.


## 1.0.0b11 (Mar. 25, 2021)

## New features

- Polyline: Add `point_along_path()` method.


## 1.0.0b10 (May 25, 2020)

### New features

- Add `Polyline.sectioned()`.
- Add stacked input support to `polliwog.tri.tri_contains_coplanar_point()`
  and `polliwog.line.coplanar_points_are_on_same_side_of_line()`.

### Optimizations

- `Polyline.segment_lengths`: Minor optimization


## 1.0.0b9 (May 7, 2020)

### New features

- Add `Plane.tilted()`.


## 1.0.0b8 (Apr. 22, 2020)

### BREAKING CHANGES

- `Polyline.points_in_front()` no longer returns points on the plane.
- Rename `polliwog.shapes.create_cube()` to `polliwog.shapes.cube`, etc.
- Remove `polliwog.shapes.create_rectangle()`.
- `polliwog.transform.apply_affine_transform` is renamed to
  `polliwog.transform.apply_transform`. It now accepts a single argument,
  a transformation matrix, which it wraps into a function which is
  returned. The function accepts a point or stack of points, along with
  two kwargs. With `discard_z_coord=True`, discard the z coordinate of
  the result. With `treat_input_as_vectors=True`, it does not use the
  homogeneous coordinate, and therefore ignores translation.

### New features

- Add `Polyline.points_on_or_in_front()`.
- Add `Polyline.path_centroid` property.
- Add `polliwog.transform.compose_transforms()`.
- Add several functions in `polliwog.transform` for orthographic viewing
  transformations.

### Other improvements

- Improve docs.


## 1.0.0b7 (Mar. 1, 2020)

### BREAKING CHANGES

- CompositeTransform, CoordinateManager: `scale()` is renamed to `uniform_scale()`.
- `polliwog.transform.transform_matrix_for_scale` is renamed to `polliwog.transform.transform_matrix_for_uniform_scale`.

### New features

- Box: Add bounding planes.
- CompositeTransform, CoordinateManager: Add `non_uniform_scale()`.
- CompositeTransform, CoordinateManager: Add `flip()`
- `polliwog.transform.transform_matrix_for_scale`: Add `allow_flipping` parameter.
- Add `polliwog.transform.transform_matrix_for_non_uniform_scale`.

### Other improvements

- Document point cloud functions.
- Omit tests from wheels.


## 1.0.0-beta.6 (Jan. 17, 2020)

### New features

- Add `polliwog.pointcloud.percentile`.


## 1.0.0-beta.5 (Jan. 13, 2020)

### BREAKING CHANGES

- Rename arguments to `polliwog.polyline.inflection_points`.

### New features

- Add `polliwog.polyline.point_of_max_acceleration`.


## 1.0.0-beta.4 (Dec. 27, 2019)

### New features

- Plane: Add `mirror_point()`.


## 1.0.0-beta.3 (Dec. 26, 2019)

### Bug fixes

- Correctly expose `mirror_point_across_plane()`.


## 1.0.0-beta.2 (Dec. 26, 2019)

### BREAKING CHANGES

- Plane: Rename `partition_by_length()` -> `subdivided_by_length()`.
- Add `mirror_point_across_plane()`.
- Remove `polliwog.transform.as_rotation_matrix()`.
- Rename `polliwog.transform.rodrigues` to `cv2_rodrigues()`. (Better to use
  one of the new functions `rodrigues_vector_to_rotation_matrix()` or
  `rotation_matrix_to_rodrigues_vector()` instead.)

## New features

- Add `rodrigues_vector_to_rotation_matrix()` and
  `rodrigues_vector_to_rotation_matrix()` as clearer versions of `rodrigues()`.

## Other maintenance

- 100% test coverage :100: :party:
- Publish the wheel using Python 3.
- Add leading underscores to all private modules, causing any
  non-canonical imports to break.


## 1.0.0-beta.1 (Dec. 5, 2019)

### BREAKING CHANGES

- Reorganize entire API into a few namespaces.
- Attach named coordinate planes to Plane class.
- Various API changes in CompositeTransform.
- Remove `CompositeTransform.append_transform3()`.
- Remove `partition_segment_old()`.
- Remove `find_rigid_rotation()` and `find_rigid_transform()`. They are
  being moved to [Entente][].
- `cut_open_polyline_by_plane()` is now private.
- Remove `estimate_normal()`.

### New features

- Break out affine transformations into their own functions.
- Plane: `sign()` and `distance()` work with single query points.
- Box: Add `contains()` method.

### Bug fixes

- Fix `contains_coplanar_point()`.

## Other maintenance

- Documentation on a single page, with sections.

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
- Publish wheels for OS X.
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

[entente]: https://github.com/lace/entente
