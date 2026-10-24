"""Batched V(s, g) from the `high_value` network, for scoring candidate goals."""
import functools

import jax
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, static_argnames='dual')
def _values(agent, observations, goals, dual):
    if dual:
        # sharsa_dual's value head reads the learned goal representation, as in its value loss.
        goals = agent.network.select('rep_value')(goals)
    v = agent.network.select('high_value')(observations, goals)
    # `high_value` is built with one ensemble member and so has no ensemble axis. Should one
    # appear, take the minimum, the pessimistic reduction SHARSA uses over its critic ensemble.
    return v.min(axis=0) if v.ndim == 2 else v


def goal_values(agent, observation, goals):
    """V(observation, g) for each row of `goals` (k, 2), as a (k,) array.

    k is padded to the next power of two, so a frontier that changes size every episode compiles
    a handful of shapes instead of one per size.
    """
    goals = np.asarray(goals, dtype=np.float32)
    k = len(goals)
    padded = 1 << max(k - 1, 0).bit_length()
    goals = np.concatenate([goals, np.repeat(goals[-1:], padded - k, axis=0)])
    observations = np.repeat(np.asarray(observation, dtype=np.float32)[None], padded, axis=0)
    dual = agent.config['agent_name'] == 'sharsa_dual'
    return np.asarray(_values(agent, jnp.asarray(observations), jnp.asarray(goals), dual))[:k]


def pair_values(agent, observations, goals):
    """V(observations[i], goals[i]) row by row, as an (n,) array."""
    dual = agent.config['agent_name'] == 'sharsa_dual'
    observations = jnp.asarray(observations, dtype=jnp.float32)
    return np.asarray(_values(agent, observations, jnp.asarray(goals, dtype=jnp.float32), dual))


@functools.partial(jax.jit, static_argnames='dual')
def _subgoal(agent, observation, goal, seed, dual):
    # The high-level half of `sample_actions`: flow-sample `num_samples` subgoals, keep the critic's argmax.
    if dual:
        goal = agent.network.select('rep_value')(goal)
    n = agent.config['num_samples']
    subgoals = jax.random.normal(seed, (n, agent.config['goal_dim']))
    observations = jnp.repeat(observation[None], n, axis=0)
    goals = jnp.repeat(goal[None], n, axis=0)
    for i in range(agent.config['flow_steps']):
        t = jnp.full((n, 1), i / agent.config['flow_steps'])
        vels = agent.network.select('high_actor_flow')(observations, goals, subgoals, t)
        subgoals = subgoals + vels / agent.config['flow_steps']
    q = agent.network.select('high_critic')(observations, goals=goals, actions=subgoals).min(axis=0)
    return subgoals[jnp.argmax(q)]


def select_subgoal(agent, observation, goal, seed):
    """The subgoal `sample_actions` would hand its low-level policy for (observation, goal), as (2,)."""
    dual = agent.config['agent_name'] == 'sharsa_dual'
    observation = jnp.asarray(observation, dtype=jnp.float32)
    goal = jnp.asarray(goal, dtype=jnp.float32)
    return np.asarray(_subgoal(agent, observation, goal, seed, dual))
