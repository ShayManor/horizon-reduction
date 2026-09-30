"""Execute one trained policy on Spot: no learning, no replay buffer, no wandb.

    python deploy/run_task.py --restore_path=<run dir> --restore_epoch=8000 --task_id=1 \
        --robot_hostname=$SPOT_IP --graphnav_map_path=spot_data --agent.w_fk=1.0 ...

The agent flags have to match the run that produced the checkpoint or the restore hits a shape
mismatch; `flags.json` in the run directory is the authoritative list.

`reset` drives Spot to the task's start waypoint under GraphNav, then the policy takes over and
drives to the goal. Positions come from `seed_tform_body`, which is the map frame the policy was
trained in, so no registration step is involved: put the robot anywhere the map covers, let it
localize, and the frames already agree.
"""
import csv
import os
import sys
import time

# This script lives in deploy/, so Python puts that directory on the path rather than the repo
# root, and the repo's own packages would not import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_online_env
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS

GOAL_DIM = 2  # the (x, y) slice the high-level policy plans over

flags.DEFINE_string('restore_path', None, 'Run directory holding params_{epoch}.pkl.')
flags.DEFINE_integer('restore_epoch', None, 'Checkpoint epoch to restore.')
flags.DEFINE_string('env_name', 'spot-maze-v0', 'Environment name.')
flags.DEFINE_string('graphnav_map_path', 'spot_data', 'Directory holding the GraphNav map.')
flags.DEFINE_string('robot_hostname', None, 'Spot address, or "sim" for the occupancy grid.')
flags.DEFINE_integer('task_id', 1, 'Task to run, 1-based, from the map task list.')
flags.DEFINE_list('task_ids', None, 'Tasks to sweep, 1-based. Overrides --task_id.')
flags.DEFINE_string('method', '', 'Label recorded in the log, e.g. phys. Does not affect the run.')
flags.DEFINE_integer('episodes', 1, 'Episodes to run back to back.')
flags.DEFINE_integer('max_episode_steps', 500, 'Truncation.')
flags.DEFINE_float('control_hz', 10.0, 'Control rate in Hz.')
flags.DEFINE_float('goal_tol', 0.5, 'Termination radius in metres.')
flags.DEFINE_list('vel_limits', ['0.6', '0.4', '0.8'], 'Forward, lateral and yaw caps.')
flags.DEFINE_float('reverse_limit', None, 'Backward cap. Defaults to the forward cap.')
flags.DEFINE_integer('seed', 0, 'Random seed for action sampling.')
flags.DEFINE_string('log_path', None, 'Optional CSV of the per-step log.')
flags.DEFINE_bool('resume', False, 'Skip task/episode pairs already in --log_path and append to it.')

config_flags.DEFINE_config_file('agent', 'agents/sharsa.py', lock_config=False)


def example_batch(env):
    """The shapes `agent.create` reads, built from the env rather than from a dataset.

    `create` touches only `observations`, `actions` and `high_actor_goals`, so a deploy run needs
    no replay buffer and no copy of the offline data on the robot.
    """
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    goals = np.zeros((1, GOAL_DIM), dtype=np.float32)
    # `sharsa` reads only the first three. `sharsa_dual` also builds its representation head from
    # `high_value_goals`, so the whole set the agents construct from is supplied here.
    return dict(
        observations=np.zeros((1, obs_dim), dtype=np.float32),
        next_observations=np.zeros((1, obs_dim), dtype=np.float32),
        actions=np.zeros((1, action_dim), dtype=np.float32),
        high_actor_goals=goals,
        high_value_goals=goals,
        low_actor_goals=goals,
        value_goals=goals,
        actor_goals=goals,
    )


def main(_):
    assert FLAGS.restore_path is not None, 'pass --restore_path'
    assert FLAGS.robot_hostname is not None, 'pass --robot_hostname (an address, or "sim")'

    env = make_online_env(
        FLAGS.env_name,
        robot_hostname=FLAGS.robot_hostname,
        graphnav_map_path=FLAGS.graphnav_map_path,
        task_ids=None,
        control_hz=FLAGS.control_hz,
        max_episode_steps=FLAGS.max_episode_steps,
        goal_tol=FLAGS.goal_tol,
        vel_limits=tuple(float(v) for v in FLAGS.vel_limits),
        reverse_limit=FLAGS.reverse_limit,
        seed=FLAGS.seed,
    )
    task_ids = [FLAGS.task_id] if FLAGS.task_ids is None else [int(t) for t in FLAGS.task_ids]

    config = FLAGS.agent
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch(env), config)
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(FLAGS.seed))

    # Written per episode rather than at the end: a run that loses the robot, the battery or the
    # container keeps everything already completed.
    # `--resume` reads back what an interrupted run already recorded and carries on from there, so
    # a benchmark split across battery swaps needs no episode arithmetic by hand.
    done = set()
    append = False
    if FLAGS.resume and FLAGS.log_path is not None and os.path.exists(FLAGS.log_path):
        with open(FLAGS.log_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row['task_id']), int(row['episode'])))
        append = True
        print(f'resuming: {len(done)} episodes already recorded in {FLAGS.log_path}')

    log = None
    writer = None
    if FLAGS.log_path is not None:
        log = open(FLAGS.log_path, 'a' if append else 'w', newline='')

    rows = []
    results = []
    try:
        for task_id in task_ids:
            task = env.task_infos[task_id - 1]
            print(f'\n=== task {task_id}: {task["task_name"]} ===')
            for episode in range(FLAGS.episodes):
                if (task_id, episode) in done:
                    continue
                ob, info = env.reset(options=dict(task_id=task_id))
                goal = info['goal']
                started = time.time()
                steps = 0
                episode_rows = []
                while True:
                    action = np.array(actor_fn(observations=ob, goals=goal))
                    ob, _, terminated, truncated, info = env.step(action)
                    steps += 1
                    row = {k: v for k, v in info.items() if k != 'goal'}
                    row['method'] = FLAGS.method
                    row['episode'] = episode
                    rows.append(row)
                    episode_rows.append(row)
                    if terminated or truncated:
                        break
                if log is not None:
                    if writer is None:
                        writer = csv.DictWriter(log, fieldnames=list(episode_rows[0].keys()))
                        if not append:
                            writer.writeheader()
                    writer.writerows(episode_rows)
                    log.flush()
                    os.fsync(log.fileno())
                outcome = 'REACHED  ' if terminated else 'TIMED OUT'
                print(f'  task {task_id} ep {episode}: {outcome} {steps:4d} steps, '
                      f'{time.time() - started:5.1f} s, final distance {info["distance_to_goal"]:.2f} m')
                results.append(dict(task=task_id, episode=episode, success=bool(terminated), steps=steps))

        print(f'\n=== summary: {FLAGS.method or "run"} ===')
        for task_id in task_ids:
            rs = [r for r in results if r['task'] == task_id]
            done = [r['steps'] for r in rs if r['success']]
            steps = f'median {sorted(done)[len(done) // 2]}' if done else 'none completed'
            print(f'  task {task_id}: {len(done)}/{len(rs)} reached, {steps}')
        total = [r for r in results if r['success']]
        print(f'  overall: {len(total)}/{len(results)}')
    finally:
        env.close()
        if log is not None:
            log.close()
            print(f'per-step log written to {FLAGS.log_path} ({len(rows)} rows)')


if __name__ == '__main__':
    app.run(main)
