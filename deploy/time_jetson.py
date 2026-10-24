"""Time one gradient update and one action per agent arm on the machine it runs on.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python deploy/time_jetson.py --out=/mnt/data/timing.json

Run it on the Jetson before choosing `--batch_size` and `--updates_per_step`: the control loop has
a 100 ms budget and the update has to fit inside it. Exits non-zero on a CPU fallback, which would
time the wrong hardware, unless `--allow_cpu` is passed.
"""
import json
import os
import runpy
import sys
import time

# This script lives in deploy/, so Python puts that directory on the path rather than the repo
# root, and the repo's own packages would not import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np
from absl import app, flags

from agents import agents

FLAGS = flags.FLAGS

OBS_DIM, GOAL_DIM, ACTION_DIM = 7, 2, 3

# The run_all.sh arms: the Spot overrides common to all, then each arm's own flags.
SPOT_OVERRIDES = dict(gc_negative=True, value_loss_type='squared', subgoal_steps=10)
ARMS = dict(
    sharsa=('agents/sharsa.py', {}),
    dual=('agents/sharsa_dual.py', dict(rep_type='bilinear', goalrep_dim=256, rep_expectile=0.9, rep_w=1.0)),
    phys=('agents/sharsa.py', dict(w_fk=1.0, viscous_scale=0.01, num_walks=10, fk_kappa=15.0)),
    geo=(
        'agents/sharsa_geodesic.py',
        dict(use_anisotropic=False, w_geo=1.0, use_fk_loss=False, basic_smoothing=False, cost_dims=2),
    ),
)

flags.DEFINE_list('agents', list(ARMS), f'Arms to time, from {list(ARMS)}.')
flags.DEFINE_list('batch_sizes', ['256', '1024'], 'Batch sizes to time.')
flags.DEFINE_integer('n', 10, 'Timed calls per measurement.')
flags.DEFINE_string('out', None, 'Where to write the JSON list of results.')
flags.DEFINE_bool('allow_cpu', False, 'Time on CPU instead of exiting.')


def synthetic_batch(batch_size):
    """Every key the four arms read during `create` and `update`, at Spot's shapes."""
    rng = np.random.default_rng(0)

    def normal(dim):
        return rng.normal(size=(batch_size, dim)).astype(np.float32)

    return dict(
        observations=normal(OBS_DIM),
        next_observations=normal(OBS_DIM),
        actions=rng.uniform(-1, 1, size=(batch_size, ACTION_DIM)).astype(np.float32),
        rewards=-np.ones(batch_size, dtype=np.float32),
        masks=np.ones(batch_size, dtype=np.float32),
        value_goals=normal(GOAL_DIM),
        actor_goals=normal(GOAL_DIM),
        high_actor_goals=normal(GOAL_DIM),
        high_actor_actions=normal(GOAL_DIM),
        high_value_reps=normal(GOAL_DIM),
        high_value_goals=normal(GOAL_DIM),
        high_value_actions=normal(GOAL_DIM),
        high_value_next_observations=normal(OBS_DIM),
        high_value_rewards=-np.ones(batch_size, dtype=np.float32),
        high_value_masks=np.ones(batch_size, dtype=np.float32),
        high_value_subgoal_steps=np.full(batch_size, 10, dtype=np.int32),
        high_value_goal_type=rng.integers(0, 3, size=batch_size).astype(np.int32),
        low_actor_goals=normal(GOAL_DIM),
    )


def build_agent(agent_file, batch_size, overrides=None):
    """Returns (agent, config) for `agent_file` under the Spot overrides plus `overrides`."""
    config = runpy.run_path(agent_file)['get_config']()
    config['goal_dim'] = GOAL_DIM
    config['action_dim'] = ACTION_DIM
    for key, value in {**SPOT_OVERRIDES, **(overrides or {})}.items():
        config[key] = value
    agent = agents[config['agent_name']].create(0, synthetic_batch(batch_size), config)
    return agent, config


def _time(fn, args, n):
    jax.block_until_ready(fn(*args))  # warm call: compile
    times = []
    for _ in range(n):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def _gflop(compiled):
    cost = compiled.cost_analysis()
    if isinstance(cost, (list, tuple)):  # older JAX returns one dict per device
        cost = cost[0] if cost else None
    return float((cost or {}).get('flops', 0.0)) / 1e9


def time_agent(agent_file, batch_size, n=10, overrides=None, name=None):
    agent, _ = build_agent(agent_file, batch_size, overrides)
    batch = synthetic_batch(batch_size)
    ob = np.zeros(OBS_DIM, dtype=np.float32)
    goal = np.ones(GOAL_DIM, dtype=np.float32)
    seed = jax.random.PRNGKey(0)

    update = jax.jit(lambda a, b: a.update(b))
    act = jax.jit(lambda a, o, g, s: a.sample_actions(o, g, s))
    update_ms, update_p95_ms = _time(update, (agent, batch), n)
    act_ms, act_p95_ms = _time(act, (agent, ob, goal, seed), n)
    return dict(
        agent=name or agent_file,
        batch_size=batch_size,
        update_ms=update_ms,
        update_p95_ms=update_p95_ms,
        act_ms=act_ms,
        act_p95_ms=act_p95_ms,
        device=str(jax.devices()[0]),
        update_gflop=_gflop(update.lower(agent, batch).compile()),
    )


def main(_):
    platform = jax.devices()[0].platform
    if platform != 'gpu' and not FLAGS.allow_cpu:
        sys.exit(f'jax is on {platform}, not gpu: this would time the wrong hardware. Pass --allow_cpu to override.')

    results = []
    for name in FLAGS.agents:
        agent_file, overrides = ARMS[name]
        for batch_size in FLAGS.batch_sizes:
            results.append(time_agent(agent_file, int(batch_size), FLAGS.n, overrides, name))
            r = results[-1]
            print(
                f"{r['agent']:>6} B={r['batch_size']:<5} update {r['update_ms']:8.1f} ms "
                f"(p95 {r['update_p95_ms']:8.1f})  act {r['act_ms']:6.1f} ms (p95 {r['act_p95_ms']:6.1f})  "
                f"{r['update_gflop']:8.1f} GFLOP  {r['device']}"
            )
    if FLAGS.out is not None:
        with open(FLAGS.out, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    app.run(main)
