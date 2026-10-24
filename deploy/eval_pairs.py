"""Ground-truth geometry for scoring a mapless checkpoint: evaluation pairs, geodesics and value goals.

Scoring only. Nothing here runs during training; it reads the sim occupancy grid as the world.
"""
import json
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from envs.spot_mapless import RunFrame


def load_start_mark(path='spot_data/start_mark.json'):
    with open(path) as f:
        mark = json.load(f)
    return RunFrame((mark['x'], mark['y'], mark['yaw']))


def save_pairs(path, pairs):
    with open(path, 'w') as f:
        json.dump(np.asarray(pairs).tolist(), f)


def geodesic(client, start_xy, goal_xy):
    """Free-space path length from start to goal, inf when the start is not free or is cut off."""
    row, col = client._cell(*start_xy)
    if not client._is_free(*start_xy):
        return math.inf
    return float(client.distance_field_xy(*goal_xy)[row, col])


def _cell_xy(client, cells):
    cells = np.asarray(cells)
    return np.stack([client.lo[0] + (cells[:, 1] + 0.5) * client.res, client.lo[1] + (cells[:, 0] + 0.5) * client.res], 1)


def _clear_cells(client, clearance, bounds=None):
    """(row, col) of free cells with EDT clearance, optionally with centres inside (xmin, ymin, xmax, ymax)."""
    ok = distance_transform_edt(client.free) * client.res >= clearance
    rows, cols = np.nonzero(ok)
    cells = np.stack([rows, cols], 1)
    if bounds is not None:
        xy = _cell_xy(client, cells)
        keep = (xy[:, 0] >= bounds[0]) & (xy[:, 1] >= bounds[1]) & (xy[:, 0] <= bounds[2]) & (xy[:, 1] <= bounds[3])
        cells = cells[keep]
    return cells


def _sample_pairs(client, cells, n, min_geo, seed):
    """Pairs of cell centres with finite geodesic >= min_geo, or None when n cannot be met."""
    rng = np.random.default_rng(seed)
    xy = _cell_xy(client, cells)
    pairs = []
    for _ in range(200 * n):
        i, j = rng.integers(len(xy), size=2)
        if min_geo <= geodesic(client, xy[i], xy[j]) < math.inf:
            pairs.append([*xy[i], *xy[j]])
            if len(pairs) == n:
                return np.array(pairs)
    return None


def make_fixed_pairs(client, bounds_seed, n=100, clearance=0.30, min_geo=1.5, seed=0):
    cells = _clear_cells(client, clearance, bounds_seed)
    pairs = _sample_pairs(client, cells, n, min_geo, seed) if len(cells) else None
    if pairs is None:
        raise RuntimeError(f'could not find {n} pairs with clearance {clearance} and geodesic >= {min_geo}')
    return pairs


def explored_pairs(buffer_path, frame, client, n=100, clearance=0.30, min_geo=1.5, seed=0):
    """Pairs between cells the buffer visited, or None when there are < 2 such cells or n cannot be met."""
    with np.load(buffer_path) as buf:
        obs = buf['observations'][:, :2]
    seed_xy = np.array([frame.from_run(x, y, 0.0)[:2] for x, y in obs])
    visited = {client._cell(x, y) for x, y in seed_xy}
    cells = _clear_cells(client, clearance)
    cells = np.array([c for c in map(tuple, cells) if c in visited]).reshape(-1, 2)
    if len(cells) < 2:
        return None
    return _sample_pairs(client, cells, n, min_geo, seed)


def value_goals(client, k=8, seed=0):
    """Farthest-point sampling over free cells with clearance >= 0.30 m, as seed-frame xy."""
    xy = _cell_xy(client, _clear_cells(client, 0.30))
    first = np.random.default_rng(seed).integers(len(xy))
    chosen = [first]
    dist = np.linalg.norm(xy - xy[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[nxt], axis=1))
    return xy[chosen]
