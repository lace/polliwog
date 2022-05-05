import numpy as np

EDGE_DTYPE = np.int64


def edges_for(num_v, is_closed):
    """
    Compute edges of a polyline, as an array of vertex indices for each edge.

    Args:
        num_v (int): The number of vertices.
        is_closed (bool): Whether or not the polyline is closed.

    Returns:
        np.ndarray: The `kx2` edges array.
    """
    num_e = num_v if is_closed else num_v - 1

    if num_e == 0:
        return np.zeros((0, 2), dtype=EDGE_DTYPE)

    edges = np.vstack(
        [np.arange(num_e, dtype=EDGE_DTYPE), np.arange(num_e, dtype=EDGE_DTYPE) + 1]
    ).T
    if is_closed:
        edges[-1][1] = 0

    return edges
