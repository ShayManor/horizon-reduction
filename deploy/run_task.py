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
import json
import math
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
from envs.spot_mapless import RunFrame, wait_for_control
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

flags.DEFINE_bool('mapless', False, 'Run the mapless pair mode: --pairs_path pairs in the seed frame, no GraphNav map.')
# score_checkpoints.py defines these two as well, and a flag can only be defined once per process.
if 'pairs_path' not in FLAGS:
    flags.DEFINE_string('pairs_path', None, 'Mapless: JSON list of [sx, sy, gx, gy] pairs in the seed frame.')
if 'start_mark' not in FLAGS:
    flags.DEFINE_string('start_mark', os.path.join(_REPO, 'spot_data', 'start_mark.json'), 'Start mark of a robot run.')
flags.DEFINE_string(
    'start_json', os.path.join(_REPO, 'spot_data', 'start.json'), 'Mapless robot: vision-frame start of the run.'
)
flags.DEFINE_list('start_pose', None, 'Mapless sim: seed-frame x,y,yaw of the start. Defaults to --start_mark.')

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


def _load_pose(path):
    with open(path) as f:
        p = json.load(f)
    return [p['x'], p['y'], p['yaw']]


def run_pairs(env, actor_fn, pairs, mark_frame, log_path, place_fn, resume=False, print_fn=print):
    """Score seed-frame [sx, sy, gx, gy] pairs, one CSV row each: outcome, steps, metres, interventions, act_ms.

    Each pair first drives to its start under the policy with the start as the goal (unscored). If that
    does not reach, `place_fn(start_seed, goal_seed)` hands the robot to the operator. The scored episode
    then runs to its goal. A lease or e-stop loss ends an episode as 'intervention' and `place_fn` follows.
    `mark_frame` takes seed-frame points to the run frame, whose origin is the pose the policy trained from.
    """
    fields = ['pair', 'sx', 'sy', 'gx', 'gy', 'outcome', 'steps', 'metres', 'interventions', 'act_ms', 'start_reached']
    done = set()
    if resume and os.path.exists(log_path):
        with open(log_path) as f:
            done = {int(r['pair']) for r in csv.DictReader(f)}
    with open(log_path, 'a' if done else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done:
            writer.writeheader()
        for i, (sx, sy, gx, gy) in enumerate(pairs):
            if i in done:
                continue
            interventions = 0

            def episode(goal_run):
                nonlocal interventions
                ob, info = env.reset(options=dict(goal=goal_run))
                goal = info['goal']
                n, metres, act = 0, 0.0, []
                while True:
                    t = time.perf_counter()
                    action = np.array(actor_fn(observations=ob, goals=goal))
                    act.append((time.perf_counter() - t) * 1000)
                    ob, _, terminated, truncated, info = env.step(action)
                    n, metres = n + 1, metres + info['step_metres']
                    if terminated or truncated:
                        break
                if info['outcome'] == 'intervention':
                    interventions += 1
                return info['outcome'], n, metres, float(np.mean(act))

            start_run = mark_frame.to_run(sx, sy, 0.0)[:2]
            goal_run = mark_frame.to_run(gx, gy, 0.0)[:2]
            outcome, *_ = episode(start_run)
            if outcome != 'reached':
                print_fn(f'pair {i}: start missed ({outcome}); place Spot at ({sx:.2f}, {sy:.2f}) in the seed frame')
                place_fn((sx, sy), (gx, gy))
            start_reached = outcome == 'reached'
            outcome, steps, metres, act_ms = episode(goal_run)
            if outcome == 'intervention':
                place_fn(None, None)
            writer.writerow(dict(pair=i, sx=sx, sy=sy, gx=gx, gy=gy, outcome=outcome, steps=steps, metres=metres,
                                 interventions=interventions, act_ms=act_ms, start_reached=start_reached))  # fmt: skip
            f.flush()
            os.fsync(f.fileno())
            print_fn(f'pair {i}: {outcome} {steps:4d} steps, {metres:5.2f} m, {interventions} interv, {act_ms:.0f} ms')


def main_mapless():
    assert FLAGS.pairs_path is not None, 'pass --pairs_path'
    assert FLAGS.log_path is not None, 'pass --log_path (the per-pair CSV; RESUME is looked for beside it)'
    sim = FLAGS.robot_hostname == 'sim'
    if sim:
        start_pose = [float(v) for v in FLAGS.start_pose] if FLAGS.start_pose else _load_pose(FLAGS.start_mark)
        origin = start_pose
    else:
        start_pose, origin = None, _load_pose(FLAGS.start_json)
    env = make_online_env(
        FLAGS.env_name,
        mapless=True,
        robot_hostname=FLAGS.robot_hostname,
        start_pose=start_pose,
        control_hz=FLAGS.control_hz,
        max_episode_steps=FLAGS.max_episode_steps,
        goal_tol=FLAGS.goal_tol,
        vel_limits=tuple(float(v) for v in FLAGS.vel_limits),
        reverse_limit=FLAGS.reverse_limit,
        seed=FLAGS.seed,
    )
    # The policy trained in the frame whose origin is the launch pose, not wherever the robot stands now.
    env.frame = RunFrame(origin)
    config = FLAGS.agent
    agent = agents[config['agent_name']].create(FLAGS.seed, example_batch(env), config)
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(FLAGS.seed))
    with open(FLAGS.pairs_path) as f:
        pairs = json.load(f)
    resume_path = os.path.join(os.path.dirname(os.path.abspath(FLAGS.log_path)), 'RESUME')

    def place(start, goal):
        if sim:  # no operator in the sim: put the robot where one would
            if start is not None:
                env.client.set_pose(*start, math.atan2(goal[1] - start[1], goal[0] - start[0]))
            return
        wait_for_control(env.client, resume_path)

    try:
        run_pairs(env, actor_fn, pairs, RunFrame(_load_pose(FLAGS.start_mark)), FLAGS.log_path, place, FLAGS.resume)
    finally:
        env.close()


def main(_):
    assert FLAGS.restore_path is not None, 'pass --restore_path'
    assert FLAGS.robot_hostname is not None, 'pass --robot_hostname (an address, or "sim")'
    if FLAGS.mapless:
        return main_mapless()

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

    # Never destroy an existing log. Without --resume the file would be reopened 'w', which is
    # how a rerun silently wipes completed episodes; move it aside instead.
    if FLAGS.log_path is not None and os.path.exists(FLAGS.log_path) and not append:
        backup = f'{FLAGS.log_path}.bak-{int(time.time())}'
        os.rename(FLAGS.log_path, backup)
        print(f'existing log moved to {backup} (pass --resume to continue it instead)')

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
