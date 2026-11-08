"""Tests for the Pure Active NTFields (`agents/ntfields.py`) and Classical Grid IPP (`agents/ipp.py`) baselines."""
import csv
import os
import subprocess
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import numpy as np
import pytest


@pytest.mark.parametrize('agent_file', ['agents/ntfields.py', 'agents/ipp.py'])
def test_build_and_update_baseline_agents(agent_file):
    from deploy.time_jetson import build_agent, synthetic_batch

    agent, config = build_agent(agent_file, 8)
    assert config['gc_negative'] and config['value_loss_type'] == 'squared' and config['subgoal_steps'] == 10
    agent, info = agent.update(synthetic_batch(8))
    assert all(np.isfinite(float(v)) for v in info.values())


@pytest.mark.parametrize('agent_file', ['agents/ntfields.py', 'agents/ipp.py'])
def test_select_subgoal_matches_sample_actions_for_baselines(agent_file):
    from deploy.time_jetson import build_agent
    from utils.value_query import select_subgoal

    agent, config = build_agent(agent_file, 8)
    ob = np.array([0.5, -0.3, 1.0, 0.0, 0.2, 0.0, 0.0], dtype=np.float32)
    goal = np.array([2.5, 1.2], dtype=np.float32)
    seed = jax.random.PRNGKey(7)
    high_seed, low_seed = jax.random.split(seed)

    subgoal = select_subgoal(agent, ob, goal, high_seed)
    assert subgoal.shape == (2,)

    actions = jax.random.normal(low_seed, (config['action_dim'],))
    for i in range(config['flow_steps']):
        t = np.full((1,), i / config['flow_steps'], dtype=np.float32)
        actions = actions + agent.network.select('low_actor_flow')(ob, subgoal, actions, t) / config['flow_steps']
    assert np.allclose(np.clip(actions, -1, 1), agent.sample_actions(ob, goal, seed), atol=1e-4)


def test_ntfields_satisfies_goal_boundary_and_speed_dependent_eikonal():
    from deploy.time_jetson import build_agent, synthetic_batch
    from utils.value_query import goal_values, pair_values

    agent, _ = build_agent('agents/ntfields.py', 8)
    ob = np.array([1.2, -0.8, 1.0, 0.0, 0.4, 0.0, 0.0], dtype=np.float32)
    same_goal = np.array([[1.2, -0.8]], dtype=np.float32)
    far_goal = np.array([[3.5, 2.0]], dtype=np.float32)

    v_same = goal_values(agent, ob, same_goal)[0]
    v_far = goal_values(agent, ob, far_goal)[0]
    assert np.isclose(v_same, 0.0, atol=1e-4)
    assert v_far < -0.1

    # Slower measured speed in observations yields higher target slowness q_s = kappa / speed.
    fast_batch = synthetic_batch(8)
    fast_batch['observations'][:, 4:6] = np.array([0.6, 0.0], dtype=np.float32)
    fast_batch['next_observations'] = fast_batch['observations'].copy()

    slow_batch = synthetic_batch(8)
    slow_batch['observations'][:, 4:6] = np.array([0.0, 0.0], dtype=np.float32)
    slow_batch['next_observations'] = slow_batch['observations'].copy()

    _, fast_info = agent.eikonal_loss(fast_batch, agent.network.params)
    _, slow_info = agent.eikonal_loss(slow_batch, agent.network.params)
    assert float(slow_info['q_s_mean']) > float(fast_info['q_s_mean']) * 5.0

    # Batch pair_values works across arbitrary pairs.
    obs_batch = np.repeat(ob[None], 4, axis=0)
    goals_batch = np.array([[1.2, -0.8], [1.5, -0.8], [2.5, -0.8], [4.0, -0.8]], dtype=np.float32)
    pv = pair_values(agent, obs_batch, goals_batch)
    assert pv.shape == (4,) and np.all(np.diff(pv) < 0)


def test_ipp_discovers_obstacles_routes_around_walls_and_persists_grid(tmp_path):
    from deploy.time_jetson import build_agent, synthetic_batch
    from utils.flax_utils import restore_agent, save_agent
    from utils.value_query import goal_values, select_subgoal

    agent, _ = build_agent('agents/ipp.py', 16)

    # Create a vertical wall of blocked contacts around x = 1.0, y in [-1.0, 1.0], from robot at x = 0.75 facing +x.
    batch = synthetic_batch(16)
    ys = np.linspace(-0.9, 0.9, 16, dtype=np.float32)
    obs = np.zeros((16, 7), dtype=np.float32)
    obs[:, 0] = 0.75
    obs[:, 1] = ys
    obs[:, 2] = 1.0  # cos(0) = 1, facing +x
    next_obs = obs.copy()  # zero displacement & zero velocity => blocked!
    actions = np.zeros((16, 3), dtype=np.float32)
    actions[:, 0] = 1.0  # full forward command into the wall

    batch['observations'] = obs
    batch['next_observations'] = next_obs
    batch['actions'] = actions

    agent, info = agent.update(batch)
    assert float(info['ipp/visited_cells']) > 0
    assert float(info['ipp/blocked_cells']) > 0

    # A goal at (2.0, 0.0) is behind the wall at x = 1.0; Dijkstra subgoal from (0.6, 0.0) must route around (|y| > 0)
    # rather than stepping straight through (1.0, 0.0).
    start_ob = np.array([0.6, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([2.0, 0.0], dtype=np.float32)
    subgoal = select_subgoal(agent, start_ob, goal, jax.random.PRNGKey(0))
    assert abs(float(subgoal[1])) > 0.1

    # Save and restore preserves the discovered grid.
    save_agent(agent, str(tmp_path), 100)
    fresh, _ = build_agent('agents/ipp.py', 16)
    restored = restore_agent(fresh, str(tmp_path), 100)
    assert np.allclose(
        restored.network.params['modules_high_value']['blocked'],
        agent.network.params['modules_high_value']['blocked'],
    )

    # goal_values scores frontier candidates using InfoGain - Dijkstra distance.
    vals = goal_values(restored, start_ob, np.array([[0.6, 1.5], [2.0, 0.0]], dtype=np.float32))
    assert vals.shape == (2,) and vals[0] > vals[1]


@pytest.mark.parametrize('agent_file', ['agents/ntfields.py', 'agents/ipp.py'])
def test_online_sim_and_score_checkpoints_for_baselines(tmp_path, agent_file):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = tmp_path / 'exp'
    cmd = [
        sys.executable,
        'main.py',
        f'--agent={agent_file}',
        '--env_name=spot-maze-v0',
        '--robot_hostname=sim',
        '--mapless',
        '--arena_bounds=-2,-2,4,4',
        '--start_pose=1,1,0',
        '--online_steps=60',
        '--warmup_steps=20',
        '--max_episode_steps=30',
        '--save_interval=60',
        '--log_interval=30',
        f'--save_dir={save_dir}',
        '--agent.batch_size=8',
        '--agent.actor_hidden_dims=(32,32)',
        '--agent.value_hidden_dims=(32,32)',
    ]
    env = dict(os.environ, WANDB_MODE='disabled', JAX_PLATFORMS='cpu')
    r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

    import glob

    (ckpt,) = glob.glob(str(save_dir / '**' / 'params_60.pkl'), recursive=True)
    run_dir = os.path.dirname(ckpt)

    score_cmd = [
        sys.executable,
        'deploy/score_checkpoints.py',
        f'--run_dir={run_dir}',
        f'--agent={agent_file}',
        '--steps=60',
        '--n_pairs=3',
        f'--pairs_path={tmp_path / "pairs.json"}',
    ]
    sr = subprocess.run(score_cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=300)
    assert sr.returncode == 0, sr.stdout[-2000:] + sr.stderr[-2000:]

    with open(os.path.join(run_dir, 'scores.csv')) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['step'] == '60'
    assert 0.0 <= float(rows[0]['fixed_SR']) <= 1.0
    assert 0.0 <= float(rows[0]['fixed_plan_SR']) <= 1.0
    assert -1.0 <= float(rows[0]['value_spearman']) <= 1.0
