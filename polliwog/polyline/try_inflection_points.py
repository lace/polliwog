def load_front_torso_mesh():
    from lace.mesh import Mesh

    mesh = Mesh(filename="/Users/pnm/code/tape/examples/anonymized_female.obj")
    mesh.cut_across_axis(1, minval=100.0, maxval=120.0)
    mesh.cut_across_axis(0, minval=-19.0, maxval=19.0)
    mesh.cut_across_axis(2, minval=0)
    return mesh


def main():
    from hobart.svg import write_polyline_3d
    import numpy as np
    from polliwog import Plane
    import vg
    from .inflection_points import inflection_points_2
    from entente.landmarks._mesh import add_landmark_points

    np.set_printoptions(suppress=False)

    mesh = load_front_torso_mesh()

    result_points = []
    for i, x_coord in enumerate(np.linspace(-15.0, 15.0, num=20)):
    #     if i != 12:
    #         continue
    # # for x_coord in [-7, 7]:
        cut_plane = Plane(np.array([x_coord, 0.0, 0.0]), vg.basis.x)
        xss = mesh.intersect_plane(cut_plane)
        longest_xs = next(reversed(sorted(xss, key=lambda xs: xs.total_length)))

        result = inflection_points_2(longest_xs.v, vg.basis.z, vg.basis.y)
        result_points.append(result)

        # if i == 12:
        #     result_points.append(result + 2.0 * vg.basis.z)

    add_landmark_points(mesh, result_points)
    mesh.write("with_inflection.dae")

    # write_polyline_3d(polyline=xs, filename="xs.svg", look=-cut_plane.normal)


if __name__ == "__main__":
    main()
