"""Draw a recorded map with its waypoint numbers, and write the task lines onto it.

    python deploy/map_tasks.py --graphnav_map_path=maps/lab
    python deploy/map_tasks.py --graphnav_map_path=maps/lab --tasks=waypoint_15:waypoint_9,...

The first call writes `waypoint_xy.json` and `plots/{map}_waypoints.png`: the lab cloud seen from
above, every waypoint labelled with its number, and the tag at the origin. Pick each task's start
and goal off that picture. The second call writes them to `tasks.json` in the order given, so the
first pair is task 1, and redraws the picture with each task as an arrow from start to goal.

Starts and goals are waypoints because `reset` drives Spot to the start under GraphNav, which can
only target a waypoint. Needs `bosdyn-api` to read the map, so run it where the SDK is installed.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

from deploy.graphnav_map import graph_distance, load_graph, load_snapshots, waypoint_poses

FLAGS = flags.FLAGS

flags.DEFINE_string('graphnav_map_path', None, 'Directory holding the recorded map.')
flags.DEFINE_list('tasks', None, 'start:goal waypoint name pairs, e.g. waypoint_15:waypoint_9.')
flags.DEFINE_string('out_dir', 'plots', 'Where the picture goes.')

RES = 0.05
# Obstacle band above the floor, and the returns a cell needs to count as occupied. This band is
# the one that keeps the lab's central island visible; higher or narrower bands wash it out.
BAND = (0.20, 1.20)
MIN_RETURNS = 2


def seed_cloud(graph, snapshots):
    """Every snapshot's point cloud in the seed frame, as an (N, 3) array."""
    from bosdyn.client.frame_helpers import get_a_tform_b
    from bosdyn.client.math_helpers import SE3Pose

    anchors = {a.id: SE3Pose.from_proto(a.seed_tform_waypoint) for a in graph.anchoring.anchors}
    clouds = []
    for waypoint in graph.waypoints:
        snapshot = snapshots.get(waypoint.snapshot_id)
        if snapshot is None or waypoint.id not in anchors:
            continue
        pc = snapshot.point_cloud
        ko_tform_sensor = get_a_tform_b(pc.source.transforms_snapshot, 'odom', pc.source.frame_name_sensor)
        seed_tform_sensor = (
            anchors[waypoint.id] * SE3Pose.from_proto(waypoint.waypoint_tform_ko) * ko_tform_sensor
        )
        points = np.frombuffer(pc.data, dtype=np.float32).reshape(-1, 3).astype(np.float64)
        rot = seed_tform_sensor.rot.to_matrix()
        clouds.append(points @ rot.T + [seed_tform_sensor.x, seed_tform_sensor.y, seed_tform_sensor.z])
    return np.concatenate(clouds)


def occupancy(cloud):
    """Cells with at least MIN_RETURNS returns in the obstacle band, and the grid's lower corner."""
    # The floor is the densest height in the cloud: every snapshot sees it, nothing else is as flat.
    counts, edges = np.histogram(cloud[:, 2], bins=np.arange(cloud[:, 2].min(), cloud[:, 2].max(), 0.02))
    floor = edges[np.argmax(counts)]
    band = cloud[(cloud[:, 2] > floor + BAND[0]) & (cloud[:, 2] < floor + BAND[1])]
    lo = np.floor(cloud[:, :2].min(axis=0) / RES) * RES
    cells = np.floor((band[:, :2] - lo) / RES).astype(int)
    shape = tuple(np.floor((cloud[:, :2].max(axis=0) - lo) / RES).astype(int)[::-1] + 1)
    hits = np.zeros(shape, dtype=int)
    np.add.at(hits, (cells[:, 1], cells[:, 0]), 1)
    return hits >= MIN_RETURNS, lo, floor


def main(_):
    assert FLAGS.graphnav_map_path is not None, 'pass --graphnav_map_path'
    map_path = FLAGS.graphnav_map_path
    graph = load_graph(map_path)
    poses = waypoint_poses(graph)
    names = {w.id: w.annotations.name for w in graph.waypoints}
    by_name = {name: wp_id for wp_id, name in names.items()}

    entries = [
        dict(id=wp_id, name=names[wp_id], x=x, y=y, yaw=yaw) for wp_id, (x, y, yaw) in poses.items()
    ]
    with open(os.path.join(map_path, 'waypoint_xy.json'), 'w') as f:
        json.dump(dict(waypoints=entries), f, indent=2)
    print(f'wrote {len(entries)} waypoints to {map_path}/waypoint_xy.json')

    tasks = []
    if FLAGS.tasks:
        for i, pair in enumerate(FLAGS.tasks):
            start, goal = pair.split(':')
            assert start in by_name and goal in by_name, f'{pair}: unknown waypoint name'
            tasks.append(dict(task_name=f'task{i + 1}', start=start, goal=goal))
        path = os.path.join(map_path, 'tasks.json')
        if os.path.exists(path):
            backup = f'{path}.bak-{int(time.time())}'
            os.rename(path, backup)
            print(f'existing tasks moved to {backup}')
        with open(path, 'w') as f:
            json.dump(tasks, f, indent=2)
        print(f'wrote {len(tasks)} tasks to {path}')
        for task in tasks:
            (sx, sy, _), (gx, gy, _) = poses[by_name[task['start']]], poses[by_name[task['goal']]]
            # GraphNav walks the graph to reach the start, so a long graph route is a slow reset.
            route = graph_distance(graph, by_name[task['start']], by_name[task['goal']])
            print(f"{task['task_name']}: {task['start']} -> {task['goal']}  "
                  f"straight {math.hypot(gx - sx, gy - sy):.2f} m, graph {route:.2f} m")

    snapshots, _ = load_snapshots(map_path, graph)
    occ, lo, floor = occupancy(seed_cloud(graph, snapshots))
    print(f'floor at z={floor:+.2f} m')

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = (lo[0], lo[0] + occ.shape[1] * RES, lo[1], lo[1] + occ.shape[0] * RES)
    ax.imshow(occ, origin='lower', extent=extent, cmap='Greys', vmax=3, interpolation='nearest')
    xy = np.array([poses[w][:2] for w in poses])
    ax.scatter(xy[:, 0], xy[:, 1], s=14, color='black', zorder=3)
    for wp_id, (x, y, _) in poses.items():
        ax.annotate(names[wp_id].replace('waypoint_', ''), (x, y), xytext=(3, 3),
                    textcoords='offset points', fontsize=8)
    ax.scatter([0], [0], marker='*', s=200, color='black', zorder=4)
    for task, color in zip(tasks, ['tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:cyan']):
        (sx, sy, _), (gx, gy, _) = poses[by_name[task['start']]], poses[by_name[task['goal']]]
        ax.annotate('', (gx, gy), (sx, sy), arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.text((sx + gx) / 2, (sy + gy) / 2, task['task_name'].replace('task', ''),
                color=color, fontsize=12, weight='bold')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    out = os.path.join(FLAGS.out_dir, f'{os.path.basename(os.path.normpath(map_path))}_waypoints.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    app.run(main)
