"""Adapter shim for the frozen `stochastic_fk_loss` (agents/fk_loss.py).

The FK loss expects:
  - self.network.select('value')(obs, goal_reps, params=...)
  - self.network.select('rep_value')(batch['value_goals'])
  - batch['observations'], batch['value_goals'], batch['speed']

This codebase uses `high_value` / `high_value_goals`, has no separate goal
encoder, and the dataset has no per-state speed. The proxies below remap the
names so the FK function runs unchanged.
"""
import jax.numpy as jnp


class _FKNetProxy:
    def __init__(self, real_network):
        self._real = real_network

    def select(self, name):
        if name == 'value':
            return self._real.select('high_value')
        if name == 'rep_value':
            def _identity(x, **_kw):
                return x
            return _identity
        return self._real.select(name)


class FKAgentProxy:
    """Stand-in for `self` inside stochastic_fk_loss."""
    def __init__(self, agent):
        self.network = _FKNetProxy(agent.network)
        self.config = agent.config


def make_fk_batch(batch, speed_source='constant'):
    """Build the small batch dict the FK loss reads.

    `speed` defaults to ones — no per-state speed in the dataset yet, so the
    Riemannian speed limit is effectively flat (q_s = kappa).

    `speed_source='observation'` instead reads the measured body speed from
    observation dims 4:6, which is where the Spot maze env puts (vx, vy). Only
    valid for that observation layout; every OGBench run uses 'constant'.
    """
    obs = batch['observations']
    if speed_source == 'observation':
        speed = jnp.linalg.norm(obs[:, 4:6], axis=-1)
    elif speed_source == 'constant':
        speed = batch.get(
            'speed',
            jnp.ones((obs.shape[0],), dtype=obs.dtype),
        )
    else:
        raise ValueError(f"unknown fk_speed_source: {speed_source}")
    return {
        'observations': obs,
        'value_goals': batch['high_value_goals'],
        'speed': speed,
    }
