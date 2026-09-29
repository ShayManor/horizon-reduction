"""Localize Spot against the map and report where it thinks it is. Nothing moves.

    python deploy/localize.py --robot_hostname=$SPOT_IP --graphnav_map_path=spot_data

Connects, uploads the map and localizes off the nearest fiducial, then prints the seed-frame pose.
It never takes the lease and never powers the robot on, so it is safe to run with Spot sitting.

The seed frame is the map frame the policy was trained in, so a successful localization here is the
whole of "same reference frame": there is no separate registration step and no need to place the
robot on any particular spot. An empty waypoint id means localization failed and every pose the
policy would see is meaningless.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app, flags

from deploy.graphnav_map import open_map
from deploy.spot_client import SpotClient
from envs.spot_maze import load_tasks

FLAGS = flags.FLAGS

flags.DEFINE_string('robot_hostname', None, 'Spot address.')
flags.DEFINE_string('graphnav_map_path', 'spot_data', 'Directory holding the GraphNav map.')
flags.DEFINE_integer('task_id', 2, 'Task whose start and goal to report distances to.')
flags.DEFINE_list('vel_limits', ['0.6', '0.4', '0.8'], 'Forward, lateral and yaw caps.')


def main(_):
    assert FLAGS.robot_hostname is not None, 'pass --robot_hostname'

    client = SpotClient(FLAGS.robot_hostname, vel_limits=tuple(float(v) for v in FLAGS.vel_limits))
    client.connect()
    print('connected, uploading map and localizing off the nearest fiducial...')
    client.upload_graph(FLAGS.graphnav_map_path)
    state = client.get_state()

    graphnav_map = open_map(FLAGS.graphnav_map_path)
    names = {}
    try:
        import json

        with open(os.path.join(FLAGS.graphnav_map_path, 'waypoint_xy.json')) as f:
            names = {e['id']: e['name'] for e in json.load(f)['waypoints']}
    except OSError:
        pass

    wp = state['localization_waypoint']
    if not wp:
        print('LOCALIZATION FAILED: no waypoint id. Move Spot where it can see the fiducial.')
        return

    x, y, yaw = state['x'], state['y'], state['yaw']
    print(f'localized at waypoint {names.get(wp, wp)}')
    print(f'seed-frame pose   x={x:+.3f}  y={y:+.3f}  yaw={math.degrees(yaw):+.1f} deg')
    print(f'distance to seed origin (0, 0): {math.hypot(x, y):.2f} m')

    nearest = min(
        graphnav_map.poses,
        key=lambda w: math.hypot(x - graphnav_map.poses[w][0], y - graphnav_map.poses[w][1]),
    )
    nx, ny, _ = graphnav_map.poses[nearest]
    print(f'nearest mapped waypoint {names.get(nearest, nearest)} at ({nx:.2f}, {ny:.2f}), '
          f'{math.hypot(x - nx, y - ny):.2f} m away')

    task = load_tasks(FLAGS.graphnav_map_path, graphnav_map)[FLAGS.task_id - 1]
    for role in ('start', 'goal'):
        tx, ty, _ = graphnav_map.poses[task[role]]
        print(f'task {FLAGS.task_id} {role:5s} {names.get(task[role], task[role])} '
              f'at ({tx:.2f}, {ty:.2f}), {math.hypot(x - tx, y - ty):.2f} m away')


if __name__ == '__main__':
    app.run(main)
