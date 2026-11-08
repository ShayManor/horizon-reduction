from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents._fk_wiring import make_fk_batch
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, MLP


class NTFieldsValue(nn.Module):
    """Factored arrival-time value V(s, g) = -||s_xy - g|| / tau(s, g) (Active NTFields).

    Satisfies V(g, g) = 0 identically and V(s, g) <= 0 everywhere, so higher V means shorter
    predicted arrival time T(s, g) = -V(s, g), matching the repo's `gc_negative=True` convention.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    tau_min: float = 0.05
    dist_eps: float = 1e-3

    def setup(self):
        self.mlp = MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals):
        goal_dim = goals.shape[-1]
        s_xy = observations[..., :goal_dim]
        dist = jnp.sqrt(jnp.sum((s_xy - goals) ** 2, axis=-1) + self.dist_eps ** 2) - self.dist_eps
        raw = self.mlp(jnp.concatenate([observations, goals], axis=-1)).squeeze(-1)
        tau = jax.nn.softplus(raw) + self.tau_min
        return -dist / tau


class NTFieldsAgent(flax.struct.PyTreeNode):
    """Pure Active NTFields baseline: first-order Eikonal PDE on V(s, g) + gradient-ascent subgoals."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def eikonal_loss(self, batch, grad_params):
        """Compute the first-order Eikonal PDE loss (||grad_xy V(s, g)|| - kappa / speed(s))^2."""
        fk_batch = make_fk_batch(batch, self.config['fk_speed_source'])
        obs = fk_batch['observations']
        goals = fk_batch['value_goals']
        raw_speed = fk_batch['speed']
        if raw_speed.ndim > 1:
            raw_speed = raw_speed.squeeze(-1)

        # Also incorporate next_observations speed so blocked transitions (where next velocity is 0)
        # supervise the slowness field at the obstacle boundary.
        if self.config['fk_speed_source'] == 'observation' and 'next_observations' in batch:
            next_speed = jnp.linalg.norm(batch['next_observations'][:, 4:6], axis=-1)
            raw_speed = jnp.minimum(raw_speed, next_speed)

        speed = jnp.clip(raw_speed, self.config['min_speed'], self.config['max_speed'])
        q_s = self.config['eikonal_kappa'] / speed

        goal_dim = goals.shape[-1]

        def _v_xy(xy, rest, goal):
            ob_full = jnp.concatenate([xy, rest], axis=-1)
            return self.network.select('high_value')(ob_full, goal, params=grad_params)

        grad_xy = jax.vmap(jax.grad(_v_xy))(obs[:, :goal_dim], obs[:, goal_dim:], goals)
        grad_norm = jnp.sqrt(jnp.sum(grad_xy ** 2, axis=-1) + 1e-8)

        v_s = self.network.select('high_value')(obs, goals, params=grad_params)
        residual = grad_norm - q_s
        loss = jnp.mean(residual ** 2)

        return loss, {
            'eikonal_loss': loss,
            'v_mean': v_s.mean(),
            'v_max': v_s.max(),
            'v_min': v_s.min(),
            'grad_norm_mean': grad_norm.mean(),
            'q_s_mean': q_s.mean(),
            'avg_speed': speed.mean(),
        }

    def low_actor_loss(self, batch, grad_params, rng):
        """Compute the low-level flow BC actor loss."""
        batch_size, action_dim = batch['actions'].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        y = x_1 - x_0

        pred = self.network.select('low_actor_flow')(
            batch['observations'], batch['low_actor_goals'], x_t, t, params=grad_params
        )

        actor_loss = jnp.mean((pred - y) ** 2)
        return actor_loss, {'actor_loss': actor_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Total loss = Eikonal PDE loss + low-level actor loss (no TD/SARSA critic or flow subgoal actor)."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, low_actor_rng = jax.random.split(rng, 2)

        eik_loss, eik_info = self.eikonal_loss(batch, grad_params)
        for k, v in eik_info.items():
            info[f'high_value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        if self.config['freeze_low_actor']:
            low_actor_loss = jax.lax.stop_gradient(low_actor_loss)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = eik_loss + low_actor_loss
        info['total_loss'] = loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the NTFields value network (and low-level actor when not frozen)."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    def _single_subgoal(self, observation, goal):
        """Steepest-ascent subgoal s_xy + delta * grad_xy V(s, g) / ||grad_xy V(s, g)||."""
        goal_dim = self.config['goal_dim']
        xy, rest = observation[:goal_dim], observation[goal_dim:]

        def _v_fn(x):
            return self.network.select('high_value')(jnp.concatenate([x, rest], axis=-1), goal)

        grad_xy = jax.grad(_v_fn)(xy)
        grad_norm = jnp.linalg.norm(grad_xy)
        diff = goal - xy
        dist_to_goal = jnp.linalg.norm(diff)
        fallback_dir = diff / (dist_to_goal + 1e-6)
        direction = jnp.where(grad_norm > 1e-6, grad_xy / (grad_norm + 1e-6), fallback_dir)
        step = jnp.minimum(self.config['subgoal_step_dist'], dist_to_goal)
        # Snap directly to goal when within one step so open-loop chains terminate inside goal_tol.
        return jnp.where(dist_to_goal <= self.config['subgoal_step_dist'], goal, xy + step * direction)

    @jax.jit
    def select_subgoal(self, observations, goals, seed=None):
        """Return the gradient-ascent subgoal for (observations, goals)."""
        if observations.ndim == 1:
            return self._single_subgoal(observations, goals)
        return jax.vmap(self._single_subgoal)(observations, goals)

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Compute gradient-ascent subgoal from V(s, g) and execute via the low-level flow policy."""
        high_seed, low_seed = jax.random.split(seed)
        subgoals = self.select_subgoal(observations, goals, high_seed)

        actions = jax.random.normal(low_seed, (*observations.shape[:-1], self.config['action_dim']))
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('low_actor_flow')(observations, subgoals, actions, t)
            actions = actions + vels / self.config['flow_steps']
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['high_actor_goals']
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]

        high_value_def = NTFieldsValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            tau_min=config['tau_min'],
        )
        low_actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )

        network_info = dict(
            high_value=(high_value_def, (ex_observations, ex_goals)),
            low_actor_flow=(low_actor_flow_def, (ex_observations, ex_goals, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['action_dim'] = action_dim
        config['goal_dim'] = goal_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='ntfields',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024, 1024, 1024, 1024),
            value_hidden_dims=(1024, 1024, 1024, 1024),
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            q_agg='min',
            action_dim=ml_collections.config_dict.placeholder(int),
            goal_dim=ml_collections.config_dict.placeholder(int),
            value_loss_type='squared',
            flow_steps=10,
            freeze_low_actor=False,
            num_samples=32,
            dataset_class='HGCDataset',
            subgoal_steps=10,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=False,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=0.5,
            actor_p_randomgoal=0.5,
            actor_geom_sample=True,
            gc_negative=True,
            # Active NTFields Eikonal parameters.
            eikonal_kappa=1.0,
            min_speed=0.05,
            max_speed=1.0,
            tau_min=0.05,
            subgoal_step_dist=1.0,
            fk_speed_source='observation',
        )
    )
    return config
