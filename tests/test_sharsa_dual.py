"""Smoke tests for SHARSADualAgent (eval-matrix method (b)).

Verifies:
1. The agent constructs without error on a tiny example batch.
2. `total_loss` returns finite values, including a non-trivial `rep/rep_contrib`.
3. One `update()` step produces finite parameters.

Mirrors the style of `tests/test_physics_loss.py` — runs on CPU with tiny
networks; no SLURM/GPU/data needed.
"""
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np

from agents.sharsa_dual import SHARSADualAgent, get_config as get_dual_config


B, D, A, G = 4, 6, 3, 6  # batch, obs-dim, action-dim, goal-dim


def make_example_batch(seed=0):
    rng = np.random.RandomState(seed)
    return {
        'observations': rng.randn(B, D).astype(np.float32),
        'next_observations': rng.randn(B, D).astype(np.float32),
        'actions': rng.randn(B, A).astype(np.float32),
        'rewards': rng.uniform(0.0, 1.0, size=(B,)).astype(np.float32),
        'masks': np.ones(B, dtype=np.float32),
        'high_actor_actions': rng.randn(B, G).astype(np.float32),
        'high_actor_goals': rng.randn(B, G).astype(np.float32),
        'high_value_goals': rng.randn(B, G).astype(np.float32),
        'high_value_actions': rng.randn(B, G).astype(np.float32),
        'high_value_next_observations': rng.randn(B, D).astype(np.float32),
        'high_value_subgoal_steps': rng.randint(1, 25, size=(B,)).astype(np.float32),
        'high_value_masks': np.ones(B, dtype=np.float32),
        'high_value_rewards': rng.uniform(0.0, 1.0, size=(B,)).astype(np.float32),
        'low_actor_goals': rng.randn(B, G).astype(np.float32),
    }


def make_dual(**overrides):
    cfg = get_dual_config()
    cfg['value_hidden_dims'] = (16, 16)
    cfg['actor_hidden_dims'] = (16, 16)
    cfg['rep_hidden_dims'] = (16, 16)
    cfg['goalrep_dim'] = 8
    for k, v in overrides.items():
        cfg[k] = v
    batch = make_example_batch()
    agent = SHARSADualAgent.create(seed=0, example_batch=batch, config=cfg)
    return agent, batch


def test_agent_constructs():
    agent, _ = make_dual()
    assert 'modules_rep_value' in agent.network.params
    assert 'modules_rep_critic' in agent.network.params
    assert 'modules_target_rep_critic' in agent.network.params


def test_total_loss_is_finite():
    agent, batch = make_dual()
    loss, info = agent.total_loss(batch, grad_params=agent.network.params, rng=jax.random.PRNGKey(1))
    assert jnp.isfinite(loss).all(), f"non-finite total loss: {loss}"
    for k, v in info.items():
        assert jnp.isfinite(v).all(), f"non-finite info[{k}] = {v}"


def test_rep_contribution_is_active():
    agent, batch = make_dual(rep_w=1.0)
    _, info = agent.total_loss(batch, grad_params=agent.network.params, rng=jax.random.PRNGKey(2))
    assert 'rep/rep_contrib' in info
    assert info['rep/rep_contrib'] != 0.0


def test_rep_w_zero_disables_rep_contribution():
    agent, batch = make_dual(rep_w=0.0)
    _, info = agent.total_loss(batch, grad_params=agent.network.params, rng=jax.random.PRNGKey(3))
    assert info['rep/rep_contrib'] == 0.0


def test_one_update_step_keeps_params_finite():
    agent, batch = make_dual()
    new_agent, info = agent.update(batch)
    leaves = jax.tree_util.tree_leaves(new_agent.network.params)
    assert all(jnp.isfinite(x).all() for x in leaves), "non-finite params after update"
    assert jnp.isfinite(info['rep/value_loss']).all()
    assert jnp.isfinite(info['rep/critic_loss']).all()


def test_target_rep_critic_lags_after_update():
    agent, batch = make_dual()
    pre_target = jax.tree_util.tree_leaves(agent.network.params['modules_target_rep_critic'])
    new_agent, _ = agent.update(batch)
    post_target = jax.tree_util.tree_leaves(new_agent.network.params['modules_target_rep_critic'])
    # tau=0.005 means target moves a tiny bit, not all the way.
    diffs = [jnp.abs(b - a).max() for a, b in zip(pre_target, post_target)]
    assert all(d < 1.0 for d in diffs), "target moved too aggressively (tau likely wrong)"
