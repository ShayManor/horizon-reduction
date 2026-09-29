"""Draw the path Spot actually walked over the lab occupancy map.

    python deploy/plot_run.py --log_path=run.csv --out=run.png

Reads the per-step CSV `run_task.py` writes and renders each episode's trajectory on the grid,
with the start waypoint, the goal and its tolerance circle marked.
"""
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from matplotlib.colors import ListedColormap

FLAGS = flags.FLAGS

flags.DEFINE_string('log_path', None, 'Per-step CSV from run_task.py.')
flags.DEFINE_string('out', 'run.png', 'Output image.')
flags.DEFINE_string('graphnav_map_path', 'spot_data', 'Directory holding occupancy.npz.')
flags.DEFINE_float('goal_tol', 0.5, 'Goal tolerance drawn as a circle.')


def episodes(rows):
    """Split the flat CSV on the step counter resetting."""
    out, cur, last = [], [], 0
    for row in rows:
        step = int(row['episode_step'])
        if step <= last and cur:
            out.append(cur)
            cur = []
        cur.append(row)
        last = step
    if cur:
        out.append(cur)
    return out


def main(_):
    assert FLAGS.log_path is not None, 'pass --log_path'
    with open(FLAGS.log_path) as f:
        rows = list(csv.DictReader(f))
    assert rows, 'the log is empty'

    grid = np.load(os.path.join(FLAGS.graphnav_map_path, 'occupancy.npz'))
    free, occ, lo, res = grid['free'], grid['occ'], grid['lo'], float(grid['res'])
    ny, nx = free.shape
    extent = [lo[0], lo[0] + nx * res, lo[1], lo[1] + ny * res]

    img = np.zeros(free.shape, dtype=int)
    img[free] = 1
    img[occ] = 2

    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.imshow(
        img, origin='lower', extent=extent, vmin=0, vmax=2, interpolation='nearest',
        cmap=ListedColormap(['#efefec', '#ffffff', '#3d3d3d']),
    )

    runs = episodes(rows)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))
    for i, (run, color) in enumerate(zip(runs, colors)):
        xs = [float(r['x']) for r in run]
        ys = [float(r['y']) for r in run]
        reached = float(run[-1]['success']) > 0
        ax.plot(xs, ys, color=color, lw=1.8, alpha=0.9,
                label=f'ep{i} {"reached" if reached else "timed out"} ({len(run)} steps)')
        ax.plot(xs[0], ys[0], 'o', color=color, ms=7)
        ax.plot(xs[-1], ys[-1], 's' if reached else 'x', color=color, ms=9, mew=2)

    with open(os.path.join(FLAGS.graphnav_map_path, 'waypoint_xy.json')) as f:
        poses = {e['id']: (e['x'], e['y']) for e in json.load(f)['waypoints']}
    gx, gy = poses[rows[0]['goal_waypoint']]
    ax.add_patch(plt.Circle((gx, gy), FLAGS.goal_tol, fill=False, color='#d1495b', lw=2))
    ax.plot(gx, gy, '*', color='#d1495b', ms=16)

    ax.set_aspect('equal')
    ax.set_xlim(-1.0, 8.0)
    ax.set_ylim(-2.2, 7.6)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(FLAGS.out, dpi=130, bbox_inches='tight')
    print(f'wrote {FLAGS.out}  ({len(runs)} episodes)')


if __name__ == '__main__':
    app.run(main)
