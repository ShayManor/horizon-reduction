"""Score mapless checkpoints in sim against the ground-truth grid (the NTFields protocol, spec §4.2).

    python deploy/score_checkpoints.py --run_dir=<run dir> --agent=agents/sharsa.py

Each `params_{step}.pkl` in the run (or each of `--steps`) becomes one row of `{run_dir}/scores.csv`.
The agent config in the run's `flags.json` wins over `--agent`, so scoring builds the network that
was trained. Run-frame poses reach the grid through the start mark: `start_pose` from `flags.json`
for a sim run, `--start_mark` for a robot run, and the identity (`--identity_frame`) for an offline
checkpoint trained in the seed frame.

Scoring only: it reads the occupancy grid as the world, which nothing that trains may do.
"""
import csv
import glob
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
from scipy.stats import spearmanr

from agents import agents
from deploy.eval_pairs import (
    _cell_xy,
    _clear_cells,
    _sample_pairs,
    explored_pairs,
    geodesic,
    load_start_mark,
    make_fixed_pairs,
    save_pairs,
    value_goals,
)
from deploy.sim_client import SimSpotClient
from envs.spot_mapless import _REPO, SIM_OCCUPANCY_PATH, MaplessSpotEnv, RunFrame
from envs.spot_maze import SIM_HOSTNAME, _no_sleep
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent
from utils.value_query import pair_values, select_subgoal

FLAGS = flags.FLAGS

# --agent comes from deploy/run_task.py, imported under __main__ below.
flags.DEFINE_string('run_dir', None, 'Run directory holding flags.json and params_{step}.pkl.')
flags.DEFINE_list('steps', None, 'Checkpoint steps to score. None scores every params_{step}.pkl.')
flags.DEFINE_bool('identity_frame', False, 'Run frame = seed frame, for offline checkpoints trained in the seed frame.')
flags.DEFINE_string('start_mark', os.path.join(_REPO, 'spot_data', 'start_mark.json'), 'Start mark of a robot run.')
flags.DEFINE_list('value_goal_ids', None, 'Waypoint names from waypoint_xy.json as value goals. None picks 8.')
flags.DEFINE_string('pairs_path', os.path.join(_REPO, 'spot_data', 'eval_pairs.json'), 'Fixed set; made if missing.')
flags.DEFINE_integer('n_pairs', 100, 'Pairs per set.')
flags.DEFINE_string('occupancy_path', SIM_OCCUPANCY_PATH, 'Ground-truth grid. waypoint_xy.json sits beside it.')

SEED = 0  # actor, chain and cell-subsample seed, so a checkpoint always scores the same
MAX_SUBGOALS = 50
CHECK_EVERY = 0.05  # metres between collision checks along the planned polyline
MAX_CELLS = 2000  # free cells the value geometry is computed over
HEADINGS = np.arange(8) * math.pi / 4  # V is averaged over these, since the metric is about position
CLEARANCE, MIN_GEO = 0.30, 1.5  # fixed-set filters
METRICS = ['SR', 'reach_rate', 'steps', 'length', 'length_ratio', 'collisions', 'plan_SR', 'plan_length', 'plan_ms']
SETS = ('fixed', 'explored')
COLUMNS = ['step', *(f'{s}_{m}' for s in SETS for m in METRICS)]
COLUMNS += ['value_spearman', 'value_spearman_h0', 'explored_skipped_reason']


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _heading(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def scoring_env(client, frame, run_flags):
    """A mapless env on `client` whose run frame is `frame`, with the run's episode settings."""
    env = MaplessSpotEnv(
        client,
        control_hz=run_flags.get('control_hz', 10.0),
        max_episode_steps=run_flags['max_episode_steps'],
        goal_tol=run_flags['goal_tol'],
        stuck_window=run_flags.get('stuck_window', 30),
        stuck_dist=run_flags.get('stuck_dist', 0.15),
        sleep_fn=_no_sleep,
    )
    # The first reset fixes the run frame at the client's pose, so put the client on the mark.
    client.set_pose(*frame.origin)
    env.reset(options={'goal': (0.0, 0.0)})
    return env


def place(env, client, frame, start, goal):
    """Start an episode at a seed-frame start, facing the seed-frame goal. Returns the observation."""
    client.set_pose(*start, _heading(start, goal))
    ob, _ = env.reset(options={'goal': frame.to_run(*goal, 0.0)[:2]})
    return ob


def rollouts(actor_fn, env, client, frame, pairs):
    """Closed-loop metrics. SR needs zero blocked steps; steps and lengths are over SR successes.

    `length_ratio` is spec-literal and biased low, equally for every arm: episodes end `goal_tol` short
    of the goal, and the geodesic is the octile grid distance to the goal cell.
    """
    reached, collisions, steps, length, ratio = [], [], [], [], []
    for sx, sy, gx, gy in pairs:
        ob = place(env, client, frame, (sx, sy), (gx, gy))
        goal = np.asarray(env.goal_xy, dtype=np.float32)
        n, blocked, metres = 0, 0, 0.0
        while True:
            ob, _, terminated, truncated, info = env.step(np.asarray(actor_fn(observations=ob, goals=goal)))
            n, blocked, metres = n + 1, blocked + info['blocked'], metres + info['step_metres']
            if terminated or truncated:
                break
        reached.append(info['outcome'] == 'reached')
        collisions.append(blocked)
        if reached[-1] and blocked == 0:
            steps.append(n)
            length.append(metres)
            ratio.append(metres / geodesic(client, (sx, sy), (gx, gy)))
    return dict(
        SR=len(steps) / len(pairs),
        reach_rate=_mean(reached),
        steps=_mean(steps),
        length=_mean(length),
        length_ratio=_mean(ratio),
        collisions=_mean(collisions),
    )


def _segment_free(client, frame, a, b):
    """Whether the run-frame segment a-b stays in free cells, checked every CHECK_EVERY metres."""
    n = max(1, math.ceil(np.linalg.norm(b - a) / CHECK_EVERY))
    for t in np.linspace(0.0, 1.0, n + 1):
        x, y, _ = frame.from_run(*(a + t * (b - a)), 0.0)
        if not client._is_free(x, y):
            return False
    return True


def planned_chain(subgoal_fn, client, frame, start, goal, goal_tol):
    """Chain high-level subgoals from a seed-frame start without executing them.

    `subgoal_fn(observation, goal)` returns a run-frame subgoal. The observation at a chain point is
    its xy at rest, facing the direction the chain arrived from (toward the goal at the start), since
    the next subgoal is not known until it is queried.
    """
    started = time.perf_counter()
    goal_run = np.array(frame.to_run(*goal, 0.0)[:2])
    points = [np.array(frame.to_run(*start, 0.0)[:2])]
    heading = _heading(points[0], goal_run)
    for _ in range(MAX_SUBGOALS):
        if np.linalg.norm(points[-1] - goal_run) <= goal_tol:
            break
        ob = np.array([*points[-1], math.cos(heading), math.sin(heading), 0.0, 0.0, 0.0], dtype=np.float32)
        nxt = np.asarray(subgoal_fn(ob, goal_run.astype(np.float32)), dtype=float)
        if not _segment_free(client, frame, points[-1], nxt):
            return dict(plan_SR=0.0, plan_length=None, plan_ms=None)
        if np.linalg.norm(nxt - points[-1]) > 0:
            heading = _heading(points[-1], nxt)
        points.append(nxt)
    if np.linalg.norm(points[-1] - goal_run) > goal_tol:
        return dict(plan_SR=0.0, plan_length=None, plan_ms=None)
    length = float(sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:])))
    return dict(plan_SR=1.0, plan_length=length, plan_ms=(time.perf_counter() - started) * 1000)


def chains(subgoal_fn, client, frame, pairs, goal_tol):
    results = [planned_chain(subgoal_fn, client, frame, p[:2], p[2:], goal_tol) for p in pairs]
    ok = [r for r in results if r['plan_SR']]
    return dict(
        plan_SR=_mean([r['plan_SR'] for r in results]),
        plan_length=_mean([r['plan_length'] for r in ok]),
        plan_ms=_mean([r['plan_ms'] for r in ok]),
    )


def value_spearman(agent, client, frame, goals_seed):
    """Spearman of V(s, g) against -geodesic over free cells, averaged over the seed-frame goals.

    Returns (headline, h0). The headline takes V at each cell as the mean over HEADINGS at rest; h0
    reads run-frame heading 0 only, the metric of the earlier offline runs, kept for continuity.
    """
    rows, cols = np.nonzero(client.free)
    if len(rows) > MAX_CELLS:
        keep = np.random.default_rng(SEED).choice(len(rows), MAX_CELLS, replace=False)
        rows, cols = rows[keep], cols[keep]
    xs, ys = client.lo[0] + (cols + 0.5) * client.res, client.lo[1] + (rows + 0.5) * client.res
    cells_run = np.array([frame.to_run(x, y, 0.0)[:2] for x, y in zip(xs, ys)])
    # One row per (cell, heading), cell-major, so a reshape to (cells, headings) undoes it.
    n, h = len(cells_run), len(HEADINGS)
    obs = np.zeros((n * h, 7), dtype=np.float32)
    obs[:, :2] = np.repeat(cells_run, h, axis=0)
    obs[:, 2], obs[:, 3] = np.tile(np.cos(HEADINGS), n), np.tile(np.sin(HEADINGS), n)
    rhos, rhos_h0 = [], []
    for gx, gy in goals_seed:
        goal = np.repeat(np.array([frame.to_run(gx, gy, 0.0)[:2]], dtype=np.float32), n * h, axis=0)
        values = pair_values(agent, obs, goal).reshape(n, h)
        geo = client.distance_field_xy(gx, gy)[rows, cols]
        ok = np.isfinite(geo)
        rhos.append(spearmanr(values[ok].mean(1), -geo[ok])[0])
        rhos_h0.append(spearmanr(values[ok, 0], -geo[ok])[0])
    return float(np.mean(rhos)), float(np.mean(rhos_h0))


def fixed_pairs(client, frame, arena_bounds, n, seed=SEED):
    """The fixed set: pairs whose start and goal both lie inside the run-frame `arena_bounds`.

    Cells are filtered in the run frame before sampling, so a start mark at any yaw keeps every pair
    inside the arena and `n` is still met. `arena_bounds` None draws from the whole grid.
    """
    if arena_bounds is None:
        return make_fixed_pairs(client, None, n=n, clearance=CLEARANCE, min_geo=MIN_GEO, seed=seed)
    xmin, ymin, xmax, ymax = (float(v) for v in arena_bounds)
    cells = _clear_cells(client, CLEARANCE)
    run = np.array([frame.to_run(x, y, 0.0)[:2] for x, y in _cell_xy(client, cells)]).reshape(-1, 2)
    inside = (run[:, 0] >= xmin) & (run[:, 1] >= ymin) & (run[:, 0] <= xmax) & (run[:, 1] <= ymax)
    pairs = _sample_pairs(client, cells[inside], n, MIN_GEO, seed) if inside.any() else None
    if pairs is None:
        raise RuntimeError(f'could not find {n} pairs inside the arena with geodesic >= {MIN_GEO}')
    return pairs


def load_agent(run_dir, step, agent_flags, env):
    # run_task registers absl flags that clash with main.py's, so it is imported only when needed.
    from deploy.run_task import example_batch

    agent = agents[agent_flags['agent_name']].create(SEED, example_batch(env), agent_flags)
    return restore_agent(agent, run_dir, step)


def score_checkpoint(run_dir, step, agent_flags, pairs_fixed, frame, client, value_goals_seed=None, n_pairs=100):
    """One row of scores.csv for `params_{step}.pkl`. Pairs and value goals are seed frame."""
    with open(os.path.join(run_dir, 'flags.json')) as f:
        run_flags = json.load(f)
    env = scoring_env(client, frame, run_flags)
    agent = load_agent(run_dir, step, agent_flags, env)

    buffer_path = os.path.join(run_dir, f'buffer_{step}.npz')
    sets, reason = {'fixed': pairs_fixed}, ''
    if not os.path.exists(buffer_path):
        reason = 'no buffer'
    else:
        sets['explored'] = explored_pairs(buffer_path, frame, client, n=n_pairs, seed=SEED)
        if sets['explored'] is None:
            reason = 'infeasible'

    row = {'step': step}
    for name in SETS:
        pairs = sets.get(name)
        if pairs is None:
            row.update({f'{name}_{m}': None for m in METRICS})
            continue
        # Fresh keys per set, so each set scores the same whatever ran before it.
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(SEED))
        subgoal_fn = supply_rng(lambda ob, g, seed: select_subgoal(agent, ob, g, seed), rng=jax.random.PRNGKey(SEED))
        metrics = rollouts(actor_fn, env, client, frame, pairs)
        metrics.update(chains(subgoal_fn, client, frame, pairs, run_flags['goal_tol']))
        row.update({f'{name}_{m}': v for m, v in metrics.items()})
    if value_goals_seed is None:
        value_goals_seed = value_goals(client, k=8, seed=SEED)
    row['value_spearman'], row['value_spearman_h0'] = value_spearman(agent, client, frame, value_goals_seed)
    row['explored_skipped_reason'] = reason
    return row


def main(_):
    assert FLAGS.run_dir is not None, 'pass --run_dir'
    with open(os.path.join(FLAGS.run_dir, 'flags.json')) as f:
        run_flags = json.load(f)
    config = FLAGS.agent.to_dict()
    config.update(run_flags['agent'])

    if FLAGS.identity_frame:
        frame = RunFrame((0.0, 0.0, 0.0))
    elif run_flags.get('robot_hostname') == SIM_HOSTNAME and run_flags.get('start_pose') is not None:
        frame = RunFrame([float(v) for v in run_flags['start_pose']])
    else:
        frame = load_start_mark(FLAGS.start_mark)

    client = SimSpotClient(
        FLAGS.occupancy_path,
        {},
        goal_tol=run_flags['goal_tol'],
        seed=SEED,
        control_hz=run_flags.get('control_hz', 10.0),
        vel_limits=tuple(float(v) for v in run_flags['vel_limits']),
        reverse_limit=run_flags.get('reverse_limit'),
    )

    # One fixed set for every arm and checkpoint: load it when it exists, else create it once.
    if os.path.exists(FLAGS.pairs_path):
        with open(FLAGS.pairs_path) as f:
            pairs_fixed = np.array(json.load(f))
        print(f'loaded {len(pairs_fixed)} fixed pairs from {FLAGS.pairs_path}')
    else:
        pairs_fixed = fixed_pairs(client, frame, run_flags.get('arena_bounds'), FLAGS.n_pairs)
        save_pairs(FLAGS.pairs_path, pairs_fixed)
        print(f'created the fixed set: {len(pairs_fixed)} pairs written to {FLAGS.pairs_path}')

    goals_seed = None
    if FLAGS.value_goal_ids is not None:
        with open(os.path.join(os.path.dirname(FLAGS.occupancy_path), 'waypoint_xy.json')) as f:
            xy = {w['name']: (w['x'], w['y']) for w in json.load(f)['waypoints']}
        goals_seed = np.array([xy[name] for name in FLAGS.value_goal_ids])

    if FLAGS.steps is not None:
        steps = [int(s) for s in FLAGS.steps]
    else:
        names = glob.glob(os.path.join(FLAGS.run_dir, 'params_*.pkl'))
        steps = sorted(int(os.path.basename(p)[len('params_') : -len('.pkl')]) for p in names)

    rows = []
    for step in steps:
        r = score_checkpoint(FLAGS.run_dir, step, config, pairs_fixed, frame, client, goals_seed, FLAGS.n_pairs)
        rows.append(r)
        explored = r['explored_skipped_reason'] or f"SR {r['explored_SR']:.2f}"
        print(
            f"step {step}: fixed SR {r['fixed_SR']:.2f}  plan_SR {r['fixed_plan_SR']:.2f}  "
            f"value_spearman {r['value_spearman']:.3f} (h0 {r['value_spearman_h0']:.3f})  explored {explored}"
        )
    path = os.path.join(FLAGS.run_dir, 'scores.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({k: '' if v is None else v for k, v in r.items()} for r in rows)
    print(f'wrote {len(rows)} rows to {path}')


if __name__ == '__main__':
    import deploy.run_task  # noqa: F401  registers --agent

    app.run(main)
