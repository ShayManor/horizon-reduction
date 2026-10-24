"""Empty-floor walking data for the low-level actor's warm start (spec section 10).

A P-controller walks the sim robot to random targets on an obstacle-free floor. The dynamics are
`SimSpotClient`'s, the observations are `MaplessSpotEnv`'s, and the file is a `ReplayBuffer.save`
buffer, so the offline Spot path (`main.py --dataset_dir`) trains on it unchanged. Nothing about the
lab is in it: no walls, no island, no start or end point.

    python -m deploy.gen_walk_data --out_dir=DIR [--n_steps=100000]
"""
import argparse
import math
import os
import tempfile

import numpy as np

from deploy.sim_client import SimSpotClient
from envs.spot_mapless import MaplessSpotEnv, RunFrame
from envs.spot_maze import _no_sleep
from utils.datasets import ReplayBuffer

GOAL_TOL = 0.25  # metres; an episode ends here, or after MAX_EPISODE_STEPS
MAX_EPISODE_STEPS = 100
P_TELEPORT = 0.1  # chance an episode starts from a random pose instead of where the last one ended
RES = 0.1


def expert_action(state, target):
    """Turn toward the target while walking forward, strafing off the lateral error. Action units."""
    dx, dy = target[0] - state['x'], target[1] - state['y']
    c, s = math.cos(state['yaw']), math.sin(state['yaw'])
    lateral = -s * dx + c * dy
    yaw_error = math.atan2(dy, dx) - state['yaw']
    yaw_error = (yaw_error + math.pi) % (2.0 * math.pi) - math.pi
    # Full speed beyond 1 m, slowing on the approach; forward speed fades as the target swings off the nose.
    speed = min(1.0, math.hypot(dx, dy))
    return np.array([
        speed * max(math.cos(yaw_error), 0.0),
        np.clip(1.5 * lateral, -1.0, 1.0) * speed,
        np.clip(1.5 * yaw_error, -1.0, 1.0),
    ])  # fmt: skip


def generate_walk_data(
    n_steps=100000,
    out_dir='walk_data',
    seed=0,
    half_extent=10.0,
    target_range=(0.5, 3.0),
    noise=0.2,
    control_hz=10.0,
    vel_limits=(0.6, 0.4, 0.8),
):
    """Write `n_steps` transitions to `out_dir/buffer_{n_steps}.npz`. Returns the written paths."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    side = int(round(2 * half_extent / RES))
    # The world is an all-free grid fed to the same sim client the online runs use. It lives in a
    # temporary directory so `--dataset_dir=out_dir` globs the buffer and nothing else.
    with tempfile.TemporaryDirectory() as tmp:
        grid = os.path.join(tmp, 'occupancy.npz')
        np.savez(grid, free=np.ones((side, side), dtype=bool), lo=np.array([-half_extent, -half_extent]), res=RES)
        client = SimSpotClient(grid, {}, goal_tol=GOAL_TOL, seed=seed, control_hz=control_hz, vel_limits=vel_limits)
    # Stuck detection is off: turning in place toward a target behind the robot is walking, not stuck.
    env = MaplessSpotEnv(
        client, control_hz=control_hz, max_episode_steps=MAX_EPISODE_STEPS, goal_tol=GOAL_TOL,
        stuck_window=MAX_EPISODE_STEPS + 1, sleep_fn=_no_sleep, frame=RunFrame((0.0, 0.0, 0.0)),
    )  # fmt: skip
    margin = half_extent - 0.5

    def random_xy():
        return rng.uniform(-margin, margin, 2)

    zeros = np.zeros(7, dtype=np.float32)
    example = dict(
        observations=zeros, next_observations=zeros, actions=np.zeros(3, dtype=np.float32),
        oracle_reps=zeros[:2], terminals=np.float32(0.0),
    )  # fmt: skip
    buffer = ReplayBuffer.create(example, n_steps + 1)  # the ring's size stops one short of its capacity
    client.set_pose(*random_xy(), rng.uniform(-math.pi, math.pi))
    while buffer.size < n_steps:
        if rng.random() < P_TELEPORT:
            client.set_pose(*random_xy(), rng.uniform(-math.pi, math.pi))
        here = np.array(client.pose[:2])
        while True:
            angle, dist = rng.uniform(-math.pi, math.pi), rng.uniform(*target_range)
            target = here + dist * np.array([math.cos(angle), math.sin(angle)])
            if np.all(np.abs(target) <= margin):
                break
        ob, _ = env.reset(options=dict(goal=target))
        done = False
        while not done and buffer.size < n_steps:
            state = client.get_state()
            action = np.clip(expert_action(state, target) + rng.normal(0.0, noise, 3), -1.0, 1.0).astype(np.float32)
            next_ob, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add_transition(dict(
                observations=ob, next_observations=next_ob, actions=action, oracle_reps=ob[:2],
                terminals=np.float32(done or buffer.size == n_steps - 1),
            ))  # fmt: skip
            ob = next_ob
    path = os.path.join(out_dir, f'buffer_{n_steps}.npz')
    buffer.save(path)
    print(f'Wrote {n_steps} empty-floor walking transitions to {path}')
    return [path]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    generate_walk_data(args.n_steps, args.out_dir, seed=args.seed)
