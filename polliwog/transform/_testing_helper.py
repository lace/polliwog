import numpy as np
from vg.compat import v2 as vg


def write_canvas_points_to_png(canvas_coords, width, height, out_path):
    import png

    vg.shape.check(locals(), "canvas_coords", (-1, 2))

    bitmap = np.zeros((height, width), dtype=np.int8)
    for x, y in canvas_coords:
        x = int(x)
        y = int(y)
        if x >= 0 and x < width and y >= 0 and y < height:
            bitmap[y][x] = 255

    png.from_array(bitmap, mode="L").save(out_path)
