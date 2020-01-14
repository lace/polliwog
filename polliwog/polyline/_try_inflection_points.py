def load_front_torso_mesh():
    from lace.mesh import Mesh

    mesh = Mesh(filename="/Users/pnm/code/tape/examples/anonymized_female.obj")
    mesh.cut_across_axis(1, minval=100.0, maxval=120.0)
    mesh.cut_across_axis(0, minval=-19.0, maxval=19.0)
    mesh.cut_across_axis(2, minval=0)
    return mesh


def main():
    # from hobart.svg import write_polyline_3d
    import numpy as np
    from polliwog import Plane
    import vg
    from .inflection_points import point_of_max_acceleration
    from entente.landmarks._mesh import add_landmark_points

    np.set_printoptions(suppress=False)

    mesh = load_front_torso_mesh()

    mid_bust = np.average(mesh.v, axis=0)
    mid_bust[2] = np.amin(mesh.v[:, 2]) + 3.0

    # to_left = vg.basis.x
    # to_center = vg.basis.z

    result_points = []
    # plot = True
    for i, x_coord in enumerate(np.linspace(-15.0, 15.0, num=20)):
        # for i, ratio in enumerate(np.linspace(0.0, 1.0, num=20)):
        # if i != 12:
        #     continue

        # mid_bust_to_bust_line = ratio * to_center + (1 - ratio) * to_left
        # cut_plane = Plane(mid_bust, vg.perpendicular(mid_bust_to_bust_line, vg.basis.y))
        cut_plane = Plane(np.array([x_coord, 0.0, 0.0]), vg.basis.x)

        xss = mesh.intersect_plane(cut_plane)
        longest_xs = next(
            reversed(sorted(xss, key=lambda xs: xs.total_length))
        ).aligned_with(vg.basis.neg_y)

        # axis = vg.normalize(np.array([0.0, -0.15, 1.0]))
        try:
            result = point_of_max_acceleration(
                # longest_xs.v, axis, vg.perpendicular(axis, vg.basis.x), span_spacing=0.001
                longest_xs.v,
                vg.basis.z,
                vg.basis.y,
                subdivide_by_length=0.001,
            )
            result_points.append(result)
        except ValueError:
            pass

    add_landmark_points(mesh, result_points)
    mesh.write("with_inflection.dae")

    # write_polyline_3d(polyline=xs, filename="xs.svg", look=-cut_plane.normal)


if __name__ == "__main__":
    main()
