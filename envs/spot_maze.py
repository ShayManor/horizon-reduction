"""Gymnasium environment for a Boston Dynamics Spot navigating a GraphNav-mapped lab maze.

The MDP:

| State       | SE(2) pose (x, y, psi)                     |
| Observation | [x, y, cos psi, sin psi, vx, vy, wz], 7    |
| Action      | body velocity [vx, vy, wz] in [-1, 1]^3    |
| Goal        | (x, y), shipped as `oracle_reps`           |
| Termination | inside `goal_tol` of the goal waypoint     |
| Truncation  | `max_episode_steps`                        |
| Reset       | GraphNav navigates back to the start       |

`x, y` are the seed-frame position GraphNav reports, and the seed frame is the map frame, so no
registration step is involved. `vx, vy, wz` are the body-frame velocity: they are not state, they
are the only readout of the controller's tracking lag.

The SDK lives behind `deploy.spot_client.SpotClient`. The env takes the client as an argument, so
everything here except `make_spot_maze_env` runs against a stand-in.
"""
import json
import math
import os
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

OBS_DIM = 7
ACTION_DIM = 3
GOAL_DIM = 2


class SpotMazeEnv(gymnasium.Env):
    """Spot navigating between GraphNav waypoint pairs.

    Args:
        client: A `SpotClient`, or anything with the same `scale_action`, `unscale_velocity`,
            `send_velocity`, `get_state`, `navigate_to`, `navigation_reached`, and
            `navigate_blocking` surface.
        waypoint_xy: {waypoint id: (x, y)} in the seed frame.
        tasks: List of {'task_name', 'start', 'goal'} dicts, `start` and `goal` being waypoint ids.
        control_hz: Control rate. One env step is one control period.
        max_episode_steps: Truncation, and therefore which episodes the steps metric excludes.
        goal_tol: Termination radius in metres.
        sleep_fn: Injection point for the control-period wait.
    """

    def __init__(
        self,
        client,
        waypoint_xy,
        tasks,
        control_hz=10.0,
        max_episode_steps=1000,
        goal_tol=0.5,
        sleep_fn=time.sleep,
    ):
        self.client = client
        self.waypoint_xy = dict(waypoint_xy)
        self.task_infos = list(tasks)
        self.control_hz = control_hz
        self.control_period = 1.0 / control_hz
        self.max_episode_steps = max_episode_steps
        self.goal_tol = goal_tol
        self._sleep = sleep_fn

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self.cur_task_id = None
        self.cur_task = None
        self.goal_xy = None
        self.step_count = 0
        self._deadline = None
        self._last_start = None

    # -- observation --------------------------------------------------------------------------

    def _observation(self, state):
        return np.array(
            [
                state['x'],
                state['y'],
                math.cos(state['yaw']),
                math.sin(state['yaw']),
                state['vx'],
                state['vy'],
                state['wz'],
            ],
            dtype=np.float32,
        )

    def _distance_to_goal(self, state):
        return math.hypot(state['x'] - self.goal_xy[0], state['y'] - self.goal_xy[1])

    def _info(self, state, success):
        info = dict(state)
        info['success'] = float(success)
        info['task_id'] = self.cur_task_id
        info['start_waypoint'] = self.cur_task['start']
        info['goal_waypoint'] = self.cur_task['goal']
        info['episode_step'] = self.step_count
        info['distance_to_goal'] = self._distance_to_goal(state)
        return info

    # -- gymnasium API ------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Drive back to the start waypoint under GraphNav and begin a new episode."""
        super().reset(seed=seed)
        options = options or {}
        task_id = options.get('task_id')
        self.cur_task_id = 1 if task_id is None else int(task_id)
        self.cur_task = self.task_infos[self.cur_task_id - 1]
        self.goal_xy = self.waypoint_xy[self.cur_task['goal']]

        self.client.navigate_blocking(self.cur_task['start'])
        self.step_count = 0
        self._last_start = time.time()
        self._deadline = self._last_start + self.control_period

        state = self.client.get_state()
        info = self._info(state, success=False)
        info['goal'] = np.array(self.goal_xy, dtype=np.float32)
        return self._observation(state), info

    def step(self, action):
        """Command one body velocity for one control period, then read the resulting state."""
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        vx, vy, wz = self.client.scale_action(action)
        self.client.send_velocity(vx, vy, wz)
        return self._advance(action, commanded=(vx, vy, wz))

    def step_graphnav(self):
        """Advance one control period under GraphNav instead of under the policy.

        Used for the seed episodes that fill the replay buffer before the first gradient step.
        NavigateTo drives the robot itself and never returns a command, so the action recorded for
        the transition is the achieved body velocity expressed back in action units.
        """
        self.client.navigate_to(self.cur_task['goal'])
        state = self.client.get_state()
        action = np.array(
            self.client.unscale_velocity(state['vx'], state['vy'], state['wz']), dtype=np.float32
        )
        return self._advance(action, commanded=(state['vx'], state['vy'], state['wz']))

    def _advance(self, action, commanded):
        """Wait out the control period, read state, and score the step."""
        now = time.time()
        if self._deadline is not None and self._deadline > now:
            self._sleep(self._deadline - now)
        started = time.time()
        # Gradient updates run between steps and can overrun the control period. `dt` is the
        # measured period, which is how that overrun becomes visible in the raw per-step log.
        dt = self.control_period if self._last_start is None else started - self._last_start
        self._last_start = started
        self._deadline = started + self.control_period

        state = self.client.get_state()
        self.step_count += 1

        terminated = self._distance_to_goal(state) <= self.goal_tol
        truncated = (not terminated) and self.step_count >= self.max_episode_steps
        if terminated or truncated:
            self.client.stop()

        info = self._info(state, success=terminated)
        info['commanded_vx'], info['commanded_vy'], info['commanded_wz'] = commanded
        info['action_x'], info['action_y'], info['action_w'] = (float(a) for a in action)
        info['dt'] = dt
        info['wall_time'] = started

        # Mirrors OGBench's single-task maze reward. Training never reads it: rewards come from
        # hindsight relabeling in GCDataset.
        reward = float(terminated) - 1.0
        return self._observation(state), reward, terminated, truncated, info

    def close(self):
        self.client.release()


def load_tasks(map_path, graphnav_map, task_ids=None):
    """Read `tasks.json` from the map directory and resolve its waypoint names to ids.

    The task list travels with the map because it is a property of the space, not of the run. Each
    entry is `{"task_name": ..., "start": ..., "goal": ...}`, where start and goal are waypoint
    ids, two-letter short codes, or annotation names.
    """
    path = os.path.join(map_path, 'tasks.json')
    assert os.path.exists(path), (
        f'{path} not found. Write the (A, B) waypoint pairs for this map there, one entry per task.'
    )
    with open(path) as f:
        raw = json.load(f)

    tasks = []
    for i, entry in enumerate(raw):
        tasks.append(
            dict(
                task_name=entry.get('task_name', f'task{i + 1}'),
                start=graphnav_map.resolve(entry['start']),
                goal=graphnav_map.resolve(entry['goal']),
            )
        )
    if task_ids is not None:
        tasks = [tasks[task_id - 1] for task_id in task_ids]
    return tasks


def make_spot_maze_env(
    env_name,
    robot_hostname,
    graphnav_map_path,
    task_ids=None,
    control_hz=10.0,
    max_episode_steps=1000,
    goal_tol=0.5,
    vel_limits=(0.6, 0.4, 0.8),
    reverse_limit=None,
):
    """Connect to Spot, upload the map, and build the env. `env_name` is accepted for symmetry."""
    from deploy.graphnav_map import GraphNavMap
    from deploy.spot_client import SpotClient

    client = SpotClient(
        robot_hostname,
        control_hz=control_hz,
        vel_limits=vel_limits,
        reverse_limit=reverse_limit,
    )
    client.connect()
    client.upload_graph(graphnav_map_path)
    client.acquire()

    graphnav_map = GraphNavMap(graphnav_map_path)
    tasks = load_tasks(graphnav_map_path, graphnav_map, task_ids)
    waypoint_xy = {wp_id: (pose[0], pose[1]) for wp_id, pose in graphnav_map.poses.items()}

    return SpotMazeEnv(
        client,
        waypoint_xy,
        tasks,
        control_hz=control_hz,
        max_episode_steps=max_episode_steps,
        goal_tol=goal_tol,
    )
