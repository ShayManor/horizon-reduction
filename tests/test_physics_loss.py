"""Tests for the physics-informed value regularization loss (paper Eq. 7).

Covers both `agents.sharsa.SHARSAAgent` and `agents.sharsa_geodesic.SHARSAGeodesicAgent`.
Runs on CPU with tiny networks; no SLURM/GPU/data needed.
"""
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.sharsa import SHARSAAgent, get_config as get_sharsa_config
from agents.sharsa_geodesic import SHARSAGeodesicAgent, get_config as get_sgeo_config


# ---------- helpers --------------------------------------------------------------

B, D, A, G = 4, 6, 3, 6  # batch, obs-dim, action-dim, goal/rep-dim (== D so 'distance' mode works)


def make_example_batch(seed=0):
    rng = np.random.RandomState(seed)
    return {
        'observations': rng.randn(B, D).astype(np.float32),
        'actions': rng.randn(B, A).astype(np.float32),
        'high_actor_actions': rng.randn(B, G).astype(np.float32),
        'high_actor_goals': rng.randn(B, G).astype(np.float32),
        'high_value_goals': rng.randn(B, G).astype(np.float32),
        'high_value_actions': rng.randn(B, G).astype(np.float32),
        'high_value_next_observations': rng.randn(B, D).astype(np.float32),
        'high_value_subgoal_steps': rng.randint(1, 25, size=(B,)).astype(np.float32),
        'high_value_masks': np.ones(B, dtype=np.float32),
        'high_value_rewards': rng.uniform(0.0, 1.0, size=(B,)).astype(np.float32),
        'high_value_goal_type': rng.randint(0, 3, size=(B,)).astype(np.int32),
        'high_value_reps': rng.randn(B, G).astype(np.float32),
        'low_actor_goals': rng.randn(B, G).astype(np.float32),
    }


def _shrink(cfg):
    cfg['value_hidden_dims'] = (16, 16)
    cfg['actor_hidden_dims'] = (16, 16)
    return cfg


def make_sgeo(**overrides):
    cfg = _shrink(get_sgeo_config())
    cfg['metric_hidden_dims'] = (16, 16)
    for k, v in overrides.items():
        cfg[k] = v
    batch = make_example_batch()
    agent = SHARSAGeodesicAgent.create(seed=0, example_batch=batch, config=cfg)
    return agent, batch


def make_sharsa(**overrides):
    cfg = _shrink(get_sharsa_config())
    for k, v in overrides.items():
        cfg[k] = v
    batch = make_example_batch()
    agent = SHARSAAgent.create(seed=0, example_batch=batch, config=cfg)
    return agent, batch


AGENT_FACTORIES = [
    pytest.param(make_sharsa, id='sharsa'),
    pytest.param(make_sgeo, id='sharsa_geodesic'),
]


# ---------- tests ----------------------------------------------------------------


@pytest.mark.parametrize('factory', AGENT_FACTORIES)
@pytest.mark.parametrize('q_mode', ['constant', 'reward', 'distance'])
def test_loss_finite_and_nonneg(factory, q_mode):
    agent, batch = factory(phy_q_mode=q_mode, phy_w=1.0)
    rng = jax.random.PRNGKey(1)
    loss, info = agent.physics_loss(batch, agent.network.params, rng)
    assert jnp.isfinite(loss)
    assert float(loss) >= 0.0
    assert jnp.isfinite(info['phy_residual_mean'])


@pytest.mark.parametrize('factory', AGENT_FACTORIES)
def test_loss_zero_when_v_constant(factory):
    """If V is identically constant in s, residual = -slack <= 0 (q,dt,nu>0) => loss=0."""
    agent, batch = factory(phy_q_mode='constant', phy_kappa=0.5, phy_nu=0.1, phy_dt=1.0)
    # Patch all 'high_value' kernel weights to zero so V output is constant in s.
    # With layer_norm=True and zero kernels at every Dense layer, every linear feature is 0,
    # then layer_norm of zeros is undefined -> we instead just zero the *final* dense's kernel.
    params = jax.tree_util.tree_map(lambda x: x, agent.network.params)
    # Find every 'kernel' under modules_high_value/value_net or similar and zero the LAST layer.
    # Simpler & more robust: set V to a constant by zeroing only the output layer kernel and bias.
    hv = params['modules_high_value']

    def find_and_zero_output(d):
        # Walk the dict; collect dense layers; zero kernel/bias of the deepest one.
        # We instead overwrite ALL Dense kernels with 0; layer_norm + 0 input is well-defined
        # (mean=0, var=0 -> out = (x-mean)/sqrt(var+eps)*scale + bias = bias). So V becomes
        # the constant produced by the final bias regardless of inputs.
        new = {}
        for k, v in d.items():
            if isinstance(v, dict):
                new[k] = find_and_zero_output(v)
            else:
                if k == 'kernel':
                    new[k] = jnp.zeros_like(v)
                else:
                    new[k] = v
        return new

    new_hv = find_and_zero_output(hv)
    new_params = dict(params)
    new_params['modules_high_value'] = new_hv
    new_params['modules_target_high_value'] = new_hv

    # Sanity: V should now be constant across the batch.
    v = agent.network.select('high_value')(batch['observations'], batch['high_value_goals'], params=new_params)
    assert jnp.allclose(v, v[0]), f'V should be constant across batch but got {v}'

    rng = jax.random.PRNGKey(2)
    loss, info = agent.physics_loss(batch, new_params, rng)
    assert float(loss) == pytest.approx(0.0, abs=1e-7)
    assert float(info['phy_violation_frac']) == 0.0


@pytest.mark.parametrize('factory', AGENT_FACTORIES)
def test_loss_active_and_grad_reduces_v_at_violation(factory):
    """Construct a violation by setting slack=0 (kappa=0) and checking that
    a loss > 0 implies grad_step decreases V(s) (since residual = V(s) - V(s')
    when V is not symmetric across the s -> s+nu*eps perturbation)."""
    agent, batch = factory(phy_q_mode='constant', phy_kappa=0.0, phy_nu=0.1, phy_dt=1.0,
                           phy_n_samples=8, phy_w=1.0)

    def loss_fn(params, rng):
        return agent.physics_loss(batch, params, rng)[0]

    rng = jax.random.PRNGKey(3)
    loss0 = float(loss_fn(agent.network.params, rng))
    if loss0 == 0.0:
        # No violation by chance; flip the test by perturbing high_value bias up.
        params = jax.tree_util.tree_map(lambda x: x, agent.network.params)
        # crude: scale all high_value Dense kernels to amplify variance => some V(s) > V(s')
        params['modules_high_value'] = jax.tree_util.tree_map(
            lambda x: x * 5.0, params['modules_high_value']
        )
        params['modules_target_high_value'] = params['modules_high_value']
        loss0 = float(loss_fn(params, rng))
        assert loss0 > 0.0, 'failed to construct a violating V'
    else:
        params = agent.network.params

    # Take a small SGD step on high_value params only, see loss go down.
    grads = jax.grad(lambda p: loss_fn(p, rng))(params)
    lr = 0.05
    new_hv = jax.tree_util.tree_map(lambda p, g: p - lr * g, params['modules_high_value'], grads['modules_high_value'])
    new_params = dict(params)
    new_params['modules_high_value'] = new_hv
    loss1 = float(loss_fn(new_params, rng))
    assert loss1 < loss0, f'expected loss to decrease, got {loss0} -> {loss1}'


@pytest.mark.parametrize('factory', AGENT_FACTORIES)
def test_target_high_value_grad_is_zero(factory):
    """Gradient of physics_loss w.r.t. target_high_value params must be exactly zero
    (target network is frozen via separate module access)."""
    agent, batch = factory(phy_q_mode='constant', phy_kappa=0.0, phy_w=1.0)
    rng = jax.random.PRNGKey(4)

    def loss_fn(params):
        return agent.physics_loss(batch, params, rng)[0]

    grads = jax.grad(loss_fn)(agent.network.params)
    target_grads = grads['modules_target_high_value']
    leaves = jax.tree_util.tree_leaves(target_grads)
    for leaf in leaves:
        assert float(jnp.abs(leaf).sum()) == 0.0, 'target_high_value should have zero grad'


@pytest.mark.parametrize('factory', AGENT_FACTORIES)
def test_phy_w_zero_does_not_alter_total_loss_versus_components(factory):
    """When phy_w=0 the physics term contributes 0 to total_loss; total should equal
    sum of the other component losses (consistency, not regression-vs-old-code)."""
    agent, batch = factory(phy_w=0.0)
    rng = jax.random.PRNGKey(5)
    total, info = agent.total_loss(batch, agent.network.params, rng=rng)

    parts = [info['high_value/value_loss'], info['high_critic/critic_loss'],
             info['high_actor/actor_loss'], info['low_actor/actor_loss']]
    if 'geodesic/reg_contrib' in info:
        parts.append(info['geodesic/reg_contrib'])
    expected = sum(float(p) for p in parts)
    assert float(total) == pytest.approx(expected, rel=1e-5, abs=1e-6)
    # And physics term contributed 0:
    assert float(info['physics/phy_contrib']) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize('factory', AGENT_FACTORIES)
def test_q_mode_dispatch(factory):
    """'reward' mode reads batch['high_value_rewards']; 'distance' reads
    batch['high_value_reps'] - batch['high_value_goals']. Probe via info['phy_q_mean']
    so the test is independent of whether V happens to violate the bound."""
    rng = jax.random.PRNGKey(6)

    def q_for(factory_fn, mode, batch):
        agent, _ = factory_fn(phy_q_mode=mode, phy_w=1.0,
                              phy_kappa=0.0, phy_nu=0.1, phy_dt=1.0)
        _, info = agent.physics_loss(batch, agent.network.params, rng)
        return float(info['phy_q_mean'])

    base_batch = make_example_batch()
    perturb_rewards = {**base_batch, 'high_value_rewards': base_batch['high_value_rewards'] * 0 + 0.99}
    perturb_reps = {**base_batch,
                    'high_value_reps': base_batch['high_value_reps'] + 10.0}

    # constant: q == kappa regardless of inputs.
    q_const_base = q_for(factory, 'constant', base_batch)
    assert q_const_base == pytest.approx(q_for(factory, 'constant', perturb_rewards), abs=1e-7)
    assert q_const_base == pytest.approx(q_for(factory, 'constant', perturb_reps), abs=1e-7)

    # reward: q sensitive to rewards, not reps.
    q_rew_base = q_for(factory, 'reward', base_batch)
    q_rew_pr = q_for(factory, 'reward', perturb_rewards)
    q_rew_pd = q_for(factory, 'reward', perturb_reps)
    assert q_rew_base != pytest.approx(q_rew_pr, abs=1e-7), 'reward mode must use rewards'
    assert q_rew_base == pytest.approx(q_rew_pd, abs=1e-7), 'reward mode must not use reps'

    # distance: q sensitive to reps, not rewards.
    q_dist_base = q_for(factory, 'distance', base_batch)
    q_dist_pr = q_for(factory, 'distance', perturb_rewards)
    q_dist_pd = q_for(factory, 'distance', perturb_reps)
    assert q_dist_base == pytest.approx(q_dist_pr, abs=1e-7), 'distance mode must not use rewards'
    assert q_dist_base != pytest.approx(q_dist_pd, abs=1e-7), 'distance mode must use reps'


@pytest.mark.parametrize('factory', AGENT_FACTORIES)
def test_full_update_runs_with_physics(factory):
    """Smoke test: a full agent.update() with phy_w>0 runs and produces finite loss,
    and target_high_value tracks high_value via EMA (changes by ~tau * delta)."""
    agent, batch = factory(phy_w=1.0, phy_q_mode='constant', phy_kappa=0.01)
    # Convert numpy batch to jnp for jit.
    jbatch = jax.tree_util.tree_map(jnp.asarray, batch)
    pre_target = jax.tree_util.tree_map(lambda x: x, agent.network.params['modules_target_high_value'])
    pre_live = jax.tree_util.tree_map(lambda x: x, agent.network.params['modules_high_value'])
    new_agent, info = agent.update(jbatch)
    assert 'physics/phy_loss' in info
    assert jnp.isfinite(info['physics/phy_loss'])
    assert jnp.isfinite(info['high_value/value_loss'])
    # target moved (EMA): post_target = tau*post_live + (1-tau)*pre_target.
    post_target = new_agent.network.params['modules_target_high_value']
    post_live = new_agent.network.params['modules_high_value']
    tau = float(agent.config['tau'])
    expected = jax.tree_util.tree_map(lambda l, pt: tau * l + (1.0 - tau) * pt, post_live, pre_target)
    diffs = jax.tree_util.tree_leaves(jax.tree_util.tree_map(
        lambda a, b: jnp.max(jnp.abs(a - b)), post_target, expected))
    max_diff = float(max(diffs)) if diffs else 0.0
    assert max_diff < 1e-5, f'EMA mismatch on target_high_value: max abs diff {max_diff}'
    # Sanity: target should not equal pre_target unless live didn't move at all.
    live_moved = float(max(jax.tree_util.tree_leaves(jax.tree_util.tree_map(
        lambda a, b: jnp.max(jnp.abs(a - b)), post_live, pre_live))))
    if live_moved > 0.0:
        target_moved = float(max(jax.tree_util.tree_leaves(jax.tree_util.tree_map(
            lambda a, b: jnp.max(jnp.abs(a - b)), post_target, pre_target))))
        assert target_moved > 0.0, 'target_high_value did not move despite live params changing'
