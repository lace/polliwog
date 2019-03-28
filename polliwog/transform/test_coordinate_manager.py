# pylint: disable=invalid-unary-operand-type
import unittest
import numpy as np
from blmath.geometry.transform.test_composite import create_cube_verts


class TestCoordinateManager(unittest.TestCase):

    def test_coordinate_manager_forward(self):
        from blmath.geometry.transform.coordinate_manager import CoordinateManager

        cube_v = create_cube_verts([1., 0., 0.], 4.)
        cube_floor_point = np.array([3., 0., 2.]) # as lace.mesh.floor_point

        coordinate_manager = CoordinateManager()
        coordinate_manager.tag_as('source')
        coordinate_manager.translate(-cube_floor_point)
        coordinate_manager.scale(2)
        coordinate_manager.tag_as('floored_and_scaled')
        coordinate_manager.translate(np.array([0., -4., 0.]))
        coordinate_manager.tag_as('centered_at_origin')

        coordinate_manager.source = cube_v

        floored_and_scaled_v = coordinate_manager.do_transform(
            cube_v,
            from_tag='source',
            to_tag='floored_and_scaled'
        )

        # Sanity check
        np.testing.assert_array_almost_equal(cube_v[0], [1., 0., 0.])
        np.testing.assert_array_almost_equal(cube_v[6], [5., 4., 4.])

        np.testing.assert_array_almost_equal(floored_and_scaled_v[0], [-4., 0., -4.])
        np.testing.assert_array_almost_equal(floored_and_scaled_v[6], [4., 8., 4.])

        centered_at_origin_v_1 = coordinate_manager.do_transform(
            cube_v,
            from_tag='source',
            to_tag='centered_at_origin'
        )
        centered_at_origin_v_2 = coordinate_manager.do_transform(
            floored_and_scaled_v,
            from_tag='floored_and_scaled',
            to_tag='centered_at_origin'
        )

        np.testing.assert_array_almost_equal(centered_at_origin_v_1[0], [-4., -4., -4.])
        np.testing.assert_array_almost_equal(centered_at_origin_v_1[6], [4., 4., 4.])

        np.testing.assert_array_almost_equal(centered_at_origin_v_2[0], [-4., -4., -4.])
        np.testing.assert_array_almost_equal(centered_at_origin_v_2[6], [4., 4., 4.])

        source_v_1 = coordinate_manager.do_transform(
            floored_and_scaled_v,
            from_tag='floored_and_scaled',
            to_tag='source'
        )
        source_v_2 = coordinate_manager.do_transform(
            centered_at_origin_v_1,
            from_tag='centered_at_origin',
            to_tag='source'
        )
        np.testing.assert_array_almost_equal(source_v_1, cube_v)
        np.testing.assert_array_almost_equal(source_v_2, cube_v)

    def test_coordinate_manager_forward_with_attrs(self):
        from blmath.geometry.transform.coordinate_manager import CoordinateManager

        cube_v = create_cube_verts([1., 0., 0.], 4.)
        cube_floor_point = np.array([3., 0., 2.]) # as lace.mesh.floor_point

        coordinate_manager = CoordinateManager()
        coordinate_manager.tag_as('source')
        coordinate_manager.translate(-cube_floor_point)
        coordinate_manager.scale(2)
        coordinate_manager.tag_as('floored_and_scaled')
        coordinate_manager.translate(np.array([0., -4., 0.]))
        coordinate_manager.tag_as('centered_at_origin')

        coordinate_manager.source = cube_v

        # Sanity check
        np.testing.assert_array_almost_equal(cube_v[0], [1., 0., 0.])
        np.testing.assert_array_almost_equal(cube_v[6], [5., 4., 4.])

        floored_and_scaled_v = coordinate_manager.floored_and_scaled
        np.testing.assert_array_almost_equal(floored_and_scaled_v[0], [-4., 0., -4.])
        np.testing.assert_array_almost_equal(floored_and_scaled_v[6], [4., 8., 4.])

        centered_at_origin_v = coordinate_manager.centered_at_origin
        np.testing.assert_array_almost_equal(centered_at_origin_v[0], [-4., -4., -4.])
        np.testing.assert_array_almost_equal(centered_at_origin_v[6], [4., 4., 4.])

        source_v = coordinate_manager.source
        np.testing.assert_array_almost_equal(source_v, cube_v)

    def test_coordinate_manager_forward_on_mesh(self):
        from mock import MagicMock
        from blmath.geometry.transform.coordinate_manager import CoordinateManager

        cube_v = create_cube_verts([1., 0., 0.], 4.)
        cube_floor_point = np.array([3., 0., 2.]) # as lace.mesh.floor_point

        # By default a magic mock will always have any attribute it's asked for;
        # here we set the spec property so that it will not respond to having a copy method
        # when the CoodinateManager looks for it.
        cube = MagicMock(spec=['v', 'other_thing'], v=cube_v, other_thing=np.array([-9.]))

        coordinate_manager = CoordinateManager()
        coordinate_manager.tag_as('source')
        coordinate_manager.translate(-cube_floor_point)
        coordinate_manager.scale(2)
        coordinate_manager.tag_as('floored_and_scaled')
        coordinate_manager.translate(np.array([0., -4., 0.]))
        coordinate_manager.tag_as('centered_at_origin')

        coordinate_manager.source = cube

        # Sanity check
        np.testing.assert_array_almost_equal(cube.v[0], [1., 0., 0.])
        np.testing.assert_array_almost_equal(cube.v[6], [5., 4., 4.])
        np.testing.assert_array_equal(cube.other_thing, [-9.])

        floored_and_scaled = coordinate_manager.floored_and_scaled
        np.testing.assert_array_almost_equal(floored_and_scaled.v[0], [-4., 0., -4.])
        np.testing.assert_array_almost_equal(floored_and_scaled.v[6], [4., 8., 4.])
        np.testing.assert_array_equal(floored_and_scaled.other_thing, [-9.])
