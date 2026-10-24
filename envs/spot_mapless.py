"""Mapless Spot pieces: the run frame, the frontier goal sampler and the reset-free env. No SDK."""
import math
import os
import time
from collections import deque

import gymnasium
import numpy as np

from envs.spot_maze import ACTION_DIM, SIM_HOSTNAME, SpotMazeEnv, _no_sleep

# The sim's world, never agent input. Resolved against the repo so the cwd does not matter.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_OCCUPANCY_PATH = os.path.join(_REPO, 'spot_data', 'occupancy.npz')

class RunFrame:
    """SE(2) pose relative to the launch pose, so the agent never sees a map frame."""

    def __init__(self, origin):
        self.origin = tuple(float(v) for v in origin)

    @classmethod
    def from_state(cls, state):
        return cls((state['x'], state['y'], state['yaw']))

    def to_run(self, x, y, yaw):
        ox, oy, oyaw = self.origin
        c, s = math.cos(oyaw), math.sin(oyaw)
        dx, dy = x - ox, y - oy
        return c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(yaw - oyaw), math.cos(yaw - oyaw))

    def from_run(self, x, y, yaw):
        ox, oy, oyaw = self.origin
        c, s = math.cos(oyaw), math.sin(oyaw)
        return ox + c * x - s * y, oy + s * x + c * y, math.atan2(math.sin(yaw + oyaw), math.cos(yaw + oyaw))


class GoalSampler:
    """Frontier goal picker over a visit grid inside a rectangle (xmin, ymin, xmax, ymax)."""

    def __init__(self, bounds, cell=0.5, min_goal_dist=1.5, p_revisit=0.3, rule='value', fail_after=2, seed=0,
                 blame_radius=1.0):
        self.xmin, self.ymin, self.xmax, self.ymax = (float(v) for v in bounds)
        xmax, ymax = self.xmax, self.ymax
        self.cell = cell
        self.min_goal_dist = min_goal_dist
        self.p_revisit = p_revisit
        self.rule = rule
        self.fail_after = fail_after
        self.blame_radius = blame_radius
        self.rng = np.random.default_rng(seed)
        self.nx = int(math.ceil((xmax - self.xmin) / cell))
        self.ny = int(math.ceil((ymax - self.ymin) / cell))
        self.visited = np.zeros((self.nx, self.ny), dtype=bool)
        self.failed = np.zeros((self.nx, self.ny), dtype=bool)
        self.fails = np.zeros((self.nx, self.ny), dtype=np.int32)

    def _index(self, x, y):
        ix, iy = math.floor((x - self.xmin) / self.cell), math.floor((y - self.ymin) / self.cell)
        return (ix, iy) if 0 <= ix < self.nx and 0 <= iy < self.ny else None

    def _centres(self, mask):
        idx = np.argwhere(mask)
        # the last row/column centre can fall past the bounds when the extent is not a multiple of cell
        x = np.minimum(self.xmin + (idx[:, 0] + 0.5) * self.cell, self.xmax)
        y = np.minimum(self.ymin + (idx[:, 1] + 0.5) * self.cell, self.ymax)
        return np.stack([x, y], 1)

    def _far(self, centres, robot_xy):
        d = np.hypot(centres[:, 0] - robot_xy[0], centres[:, 1] - robot_xy[1])
        return centres[d >= self.min_goal_dist]

    def observe(self, x, y):
        i = self._index(x, y)
        if i is not None:
            self.visited[i] = True

    def _frontier_mask(self):
        pad = np.pad(self.visited, 1)
        near = np.zeros_like(self.visited)
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                if (dx, dy) != (1, 1):
                    near |= pad[dx:dx + self.nx, dy:dy + self.ny]
        return near & ~self.visited & ~self.failed

    def frontier(self, robot_xy):
        return self._far(self._centres(self._frontier_mask()), robot_xy)

    def sample(self, robot_xy, value_fn):
        revisit = self._far(self._centres(self.visited & ~self.failed), robot_xy)
        front = self.frontier(robot_xy)
        if self.rng.random() < self.p_revisit:
            first, second = revisit, front
        else:
            first, second = front, revisit
        for pool in (first, second):
            if len(pool) == 0:
                continue
            if pool is front:
                if self.rule == 'nearest':
                    return pool[np.argmin(np.hypot(pool[:, 0] - robot_xy[0], pool[:, 1] - robot_xy[1]))]
                return pool[int(np.argmax(value_fn(pool)))]
            return pool[self.rng.integers(len(pool))]
        free = self._centres(~self.failed)
        if len(free) == 0:
            free = self._centres(np.ones_like(self.failed))
        far = self._far(free, robot_xy)
        if len(far) == 0:
            # nothing eligible is min_goal_dist away: the farthest cell, never the robot's own
            return free[np.argmax(np.hypot(free[:, 0] - robot_xy[0], free[:, 1] - robot_xy[1]))]
        return far[self.rng.integers(len(far))]

    def report(self, goal_xy, outcome, blocked, warmup, end_xy=None):
        i = self._index(goal_xy[0], goal_xy[1])
        if i is None or outcome != 'stuck' or not blocked or warmup:
            return
        # a robot pinned elsewhere says nothing about this goal cell
        if end_xy is not None and math.hypot(end_xy[0] - goal_xy[0], end_xy[1] - goal_xy[1]) > self.blame_radius:
            return
        self.fails[i] += 1
        if self.fails[i] >= self.fail_after:
            self.failed[i] = True

    def coverage(self):
        return float(np.mean(self.visited))

    def state_dict(self):
        return {'visited': self.visited.tolist(), 'failed': self.failed.tolist(), 'fails': self.fails.tolist()}

    def load_state_dict(self, d):
        self.visited = np.asarray(d['visited'], dtype=bool).reshape(self.nx, self.ny)
        self.failed = np.asarray(d['failed'], dtype=bool).reshape(self.nx, self.ny)
        self.fails = np.asarray(d['fails'], dtype=np.int32).reshape(self.nx, self.ny)

    def stats(self):
        return {
            'frontier_size': int(self._frontier_mask().sum()),
            'failed_cells': int(self.failed.sum()),
            'visited_cells': int(self.visited.sum()),
        }


class ABGoals:
    """Alternating end/start goals in the tag frame, behind the `GoalSampler` interface.

    The first goal is end. Only 'reached' switches it: stuck, timeout and intervention keep it, and
    the next attempt starts from wherever the robot stands.
    """

    names = ('end', 'start')

    def __init__(self, end_xy, start_xy):
        self.goals = (tuple(float(v) for v in end_xy[:2]), tuple(float(v) for v in start_xy[:2]))
        self.idx = 0

    def sample(self, robot_xy, value_fn):
        return np.array(self.goals[self.idx], dtype=np.float32)

    def observe(self, x, y):
        pass

    def report(self, goal_xy, outcome, blocked, warmup, end_xy=None):
        if outcome == 'reached':
            self.idx = 1 - self.idx

    def coverage(self):
        return 0.0

    def stats(self):
        return {'frontier_size': 0, 'failed_cells': 0, 'visited_cells': 0}

    def state_dict(self):
        return {'goal_idx': self.idx}

    def load_state_dict(self, d):
        self.idx = int(d['goal_idx'])


def is_blocked(commanded_xy, measured_xy):
    """Collision proxy on the robot: a planar command above 0.2 m/s that moves the body under 0.05 m/s."""
    return math.hypot(*commanded_xy) > 0.2 and math.hypot(*measured_xy) < 0.05


def _is_intervention(exc):
    # Matched by name so this module never imports bosdyn: LeaseUseError, EstopStoppedError, etc.
    # A command sent after a hardware e-stop cut power raises NotPoweredOnError.
    name = type(exc).__name__
    return any(k in name for k in ('Lease', 'Estop', 'EStop', 'NotPoweredOn'))


def wait_for_control(client, resume_path, sleep_fn=time.sleep, print_fn=print, poll=1.0, report_every=10.0):
    """Block after a lease or e-stop intervention until the operator creates `resume_path`, then take control.

    The file is the operator's consent: a reachable robot is not one handed back, and `acquire` takes
    the lease from whoever holds it. Polls every `poll` seconds and prints one line per `report_every`.
    The file is deleted, the old keepalives are shut down so retries do not leak them, and `acquire`
    runs once.
    """
    waited, next_report = 0.0, 0.0
    while not os.path.exists(resume_path):
        if waited >= next_report:
            print_fn(f'waiting for control ({waited:.0f} s): create {resume_path} once Spot is handed back')
            next_report += report_every
        sleep_fn(poll)
        waited += poll
    os.remove(resume_path)
    for name in ('_lease_keepalive', '_estop_keepalive'):
        keepalive = getattr(client, name, None)
        if keepalive is not None:
            keepalive.shutdown()
            setattr(client, name, None)
    client.acquire()


class MaplessSpotEnv(SpotMazeEnv):
    """Reset-free Spot navigation in the run frame. One episode is one goal attempt.

    The first `reset` fixes the run frame at the current pose, unless `frame` (a RunFrame, e.g. the
    AprilTag's) is given; later resets keep it and start from wherever the robot stands. Episodes end as 'reached', 'timeout', 'stuck' (under `stuck_dist` of
    net displacement over `stuck_window` steps), or 'intervention' (lease or e-stop lost, or a pose
    jump over `jump_dist` in one step, which is odometry resetting or the robot being carried).
    """

    def __init__(
        self,
        client,
        control_hz=10.0,
        max_episode_steps=300,
        goal_tol=0.5,
        stuck_window=30,
        stuck_dist=0.15,
        jump_dist=1.0,
        sleep_fn=time.sleep,
        frame=None,
    ):
        super().__init__(
            client,
            {},
            [],
            control_hz=control_hz,
            max_episode_steps=max_episode_steps,
            goal_tol=goal_tol,
            sleep_fn=sleep_fn,
        )
        self.stuck_dist = stuck_dist
        self.jump_dist = jump_dist
        self.frame = frame
        # The reset pose plus `stuck_window` steps, so the window spans `stuck_window` intervals.
        self._recent = deque(maxlen=stuck_window + 1)
        self._last_state = None

    def _run_state(self, state):
        state = dict(state)
        state['x'], state['y'], state['yaw'] = self.frame.to_run(state['x'], state['y'], state['yaw'])
        return state

    def _distance_to_goal(self, state):
        return math.hypot(state['x'] - self.goal_xy[0], state['y'] - self.goal_xy[1])

    def _info(self, state, outcome, blocked=False, step_metres=0.0):
        info = dict(state)
        info['outcome'] = outcome
        info['success'] = float(outcome == 'reached')
        info['blocked'] = bool(blocked)
        info['step_metres'] = float(step_metres)
        info['episode_step'] = self.step_count
        info['distance_to_goal'] = self._distance_to_goal(state)
        info['goal'] = np.array(self.goal_xy, dtype=np.float32)
        return info

    def reset(self, seed=None, options=None):
        """Start a goal attempt from the current pose. `options['goal']` is run-frame (x, y)."""
        gymnasium.Env.reset(self, seed=seed)
        self.goal_xy = tuple(float(v) for v in options['goal'])
        state = self.client.get_state()
        if self.frame is None:
            self.frame = RunFrame.from_state(state)
        state = self._run_state(state)
        self.step_count = 0
        self._recent.clear()
        self._recent.append((state['x'], state['y']))
        self._last_state = state
        self._last_start = time.time()
        self._deadline = self._last_start + self.control_period
        return self._observation(state), self._info(state, 'running')

    def _wait(self):
        """The control-period wait of `SpotMazeEnv._advance`. Returns the measured period and its start."""
        now = time.time()
        if self._deadline is not None and self._deadline > now:
            self._sleep(self._deadline - now)
        started = time.time()
        dt = self.control_period if self._last_start is None else started - self._last_start
        self._last_start = started
        self._deadline = started + self.control_period
        return dt, started

    def _stop(self):
        try:
            self.client.stop()
        except Exception as exc:  # stopping without the lease raises the same error again
            if not _is_intervention(exc):
                raise

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        vx, vy, wz = self.client.scale_action(action)
        dt, started, raw = self.control_period, time.time(), None
        try:
            self.client.send_velocity(vx, vy, wz)
            dt, started = self._wait()
            raw = self.client.get_state()
        except Exception as exc:  # only lease and e-stop errors are caught
            if not _is_intervention(exc):
                raise
        self.step_count += 1

        blocked, step_metres, jump = False, 0.0, False
        if raw is None:
            outcome, state = 'intervention', self._last_state
        else:
            if hasattr(self.client, 'blocked'):
                blocked = self.client.blocked
            else:
                blocked = is_blocked((vx, vy), (raw['vx'], raw['vy']))
            state = self._run_state(raw)
            step_metres = math.hypot(state['x'] - self._last_state['x'], state['y'] - self._last_state['y'])
            self._recent.append((state['x'], state['y']))
            oldest, newest = self._recent[0], self._recent[-1]
            if step_metres > self.jump_dist:
                outcome, step_metres, jump = 'intervention', 0.0, True
            elif self._distance_to_goal(state) <= self.goal_tol:
                outcome = 'reached'
            elif (
                len(self._recent) == self._recent.maxlen
                and math.hypot(newest[0] - oldest[0], newest[1] - oldest[1]) < self.stuck_dist
            ):
                outcome = 'stuck'
            elif self.step_count >= self.max_episode_steps:
                outcome = 'timeout'
            else:
                outcome = 'running'
        self._last_state = state

        terminated = outcome == 'reached'
        truncated = outcome not in ('running', 'reached')
        if terminated or truncated:
            self._stop()

        info = self._info(state, outcome, blocked, step_metres)
        info['commanded_vx'], info['commanded_vy'], info['commanded_wz'] = vx, vy, wz
        info['action_x'], info['action_y'], info['action_w'] = (float(a) for a in action)
        info['dt'] = dt
        info['wall_time'] = started
        info['jump'] = jump
        reward = float(terminated) - 1.0
        return self._observation(state), reward, terminated, truncated, info


def make_spot_mapless_env(
    robot_hostname,
    start_pose=None,
    control_hz=10.0,
    max_episode_steps=300,
    goal_tol=0.5,
    vel_limits=(0.6, 0.4, 0.8),
    reverse_limit=None,
    seed=0,
    stuck_window=30,
    stuck_dist=0.15,
    frame=None,
):
    """Build the mapless env. No GraphNav map and no task list, on the robot or in the sim.

    The sim reads `spot_data/occupancy.npz` as its world and starts at `start_pose` (seed frame).
    The robot reads its pose from visual-inertial odometry and ignores `start_pose`.
    """
    if robot_hostname == SIM_HOSTNAME:
        from deploy.sim_client import SimSpotClient

        assert start_pose is not None, 'the mapless sim needs start_pose (x, y, yaw) in the seed frame'
        client = SimSpotClient(
            SIM_OCCUPANCY_PATH,
            {},
            goal_tol=goal_tol,
            seed=seed,
            control_hz=control_hz,
            vel_limits=vel_limits,
            reverse_limit=reverse_limit,
        )
        client.set_pose(*start_pose)
        sleep_fn = _no_sleep
    else:
        from deploy.spot_client import SpotClient

        client = SpotClient(
            robot_hostname,
            pose_source='vision',
            control_hz=control_hz,
            vel_limits=vel_limits,
            reverse_limit=reverse_limit,
        )
        client.connect()
        client.acquire()
        sleep_fn = time.sleep

    return MaplessSpotEnv(
        client,
        control_hz=control_hz,
        max_episode_steps=max_episode_steps,
        goal_tol=goal_tol,
        stuck_window=stuck_window,
        stuck_dist=stuck_dist,
        sleep_fn=sleep_fn,
        frame=frame,
    )
