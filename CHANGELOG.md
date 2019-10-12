# Changelog

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
