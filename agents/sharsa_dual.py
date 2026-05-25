"""SHARSA + dual goal representation (eval-matrix method (b)).

Drops in the dual paper's *auxiliary* representation loss — `rep_loss` (IQL
value+critic on a learned goal rep) — and routes high-level value/critic/actor
through `rep_value(g)` instead of the raw goal observation.

What stays from upstream `agents/sharsa.py` (unchanged):
  - hierarchical flow-BC actors (high + low)
  - SARSA-style high-level critic
  - target updates for `high_value` and `high_critic`

What's new from the dual paper (`dual-goal-viscosity/agents/gcivl/state/dual.py`):
  - `rep_value` head: bilinear DualRepresentationValue producing psi(g).
  - `rep_critic` ensemble + `target_rep_critic`: auxiliary IQL critic.
  - `rep_loss`: expectile value loss + TD critic loss on the rep.
  - Goals fed to high-level networks pass through `rep_value(g)` first.

Note: `rep_value` is called *without* `params=grad_params` everywhere except
inside `rep_loss` itself — so gradient flows into the rep only via `rep_loss`,
matching dual.py. SHARSA's high-level losses see the rep as a fixed projection.

Phys/FK paths are inherited from `sharsa.py` but disabled by default
(`phy_w=0, w_fk=0`) so this row isolates the dual-rep contribution.
"""
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.dual import DualRepresentationValue
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCValue


class SHARSADualAgent(flax.struct.PyTreeNode):
    """SHARSA agent with a learned dual goal representation (paper P2)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        return -(log_pred * target + log_not_pred * (1 - target))

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Asymmetric L2 from IQL — used by rep_loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    # ------- dual-paper auxiliary loss --------------------------------------

    def rep_loss(self, batch, grad_params):
        """IQL value+critic on the dual rep. Ported from dual.py:111-141 with
        sigmoid bounding for stability under SHARSA reward semantics.

        - rep_value (V_phi(s, g)) trained with expectile regression toward
          rep_critic's min Q. Both wrapped in sigmoid before the loss so
          everything stays in [0,1]; matches SHARSA's value_loss_type='bce'
          stabilization. Without sigmoid the IQL loop diverges under
          gc_negative=False + discount=0.999 (the companion repo avoids this
          via gc_negative=True + discount=0.99; we can't switch globally
          without breaking SHARSA's BCE high-level head).
        - rep_critic trained with squared TD where the bootstrap is
          sigmoid(V_phi(s', g)). With rewards in {0,1} and `mask=0 at goal`,
          the TD target stays in [0,1], so squared loss against sigmoid'd Q
          is well-behaved.

        HGCDataset doesn't emit `batch['value_goals']` (only `high_value_goals`),
        so we read the high-level goal — same goal-sampling distribution, just a
        different key name in the hierarchical dataset.
        """
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['high_value_goals']
        actions = batch['actions']

        # Sigmoid-bound rep value + critic. Logits are kept for diagnostics only.
        q1_logit, q2_logit = self.network.select('target_rep_critic')(obs, goals, actions)
        q1_t, q2_t = jax.nn.sigmoid(q1_logit), jax.nn.sigmoid(q2_logit)
        q_min = jnp.minimum(q1_t, q2_t)
        v_logit = self.network.select('rep_value')(obs, goals, params=grad_params)
        v = jax.nn.sigmoid(v_logit)
        value_loss = self.expectile_loss(q_min - v, q_min - v, self.config['rep_expectile']).mean()

        next_v_logit = self.network.select('rep_value')(next_obs, goals)
        next_v = jax.nn.sigmoid(next_v_logit)
        td_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1_pred_logit, q2_pred_logit = self.network.select('rep_critic')(
            obs, goals, actions, params=grad_params
        )
        q1_pred = jax.nn.sigmoid(q1_pred_logit)
        q2_pred = jax.nn.sigmoid(q2_pred_logit)
        critic_loss = ((q1_pred - td_target) ** 2 + (q2_pred - td_target) ** 2).mean()

        total = value_loss + critic_loss
        return total, {
            'value_loss': value_loss,
            'critic_loss': critic_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'v_logit_abs_mean': jnp.abs(v_logit).mean(),
            'q_mean': q1_pred.mean(),
            'td_target_mean': td_target.mean(),
            'td_target_max': td_target.max(),
        }

    # ------- SHARSA losses, with goals routed through rep_value -------------

    def high_value_loss(self, batch, grad_params):
        goal_reps = self.network.select('rep_value')(batch['high_value_goals'])
        q1, q2 = self.network.select('target_high_critic')(
            batch['observations'], goals=goal_reps, actions=batch['high_value_actions']
        )
        if self.config['value_loss_type'] == 'bce':
            q1, q2 = jax.nn.sigmoid(q1), jax.nn.sigmoid(q2)
        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        elif self.config['q_agg'] == 'mean':
            q = (q1 + q2) / 2

        v = self.network.select('high_value')(batch['observations'], goal_reps, params=grad_params)
        if self.config['value_loss_type'] == 'bce':
            v_logit = v
            v = jax.nn.sigmoid(v_logit)

        if self.config['value_loss_type'] == 'squared':
            value_loss = ((v - q) ** 2).mean()
        elif self.config['value_loss_type'] == 'bce':
            value_loss = self.bce_loss(v_logit, q).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def high_critic_loss(self, batch, grad_params):
        goal_reps = self.network.select('rep_value')(batch['high_value_goals'])
        next_v = self.network.select('high_value')(batch['high_value_next_observations'], goal_reps)
        if self.config['value_loss_type'] == 'bce':
            next_v = jax.nn.sigmoid(next_v)
        q = (
            batch['high_value_rewards']
            + (self.config['discount'] ** batch['high_value_subgoal_steps']) * batch['high_value_masks'] * next_v
        )

        q1, q2 = self.network.select('high_critic')(
            batch['observations'], goal_reps, batch['high_value_actions'], params=grad_params
        )

        if self.config['value_loss_type'] == 'squared':
            critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        elif self.config['value_loss_type'] == 'bce':
            q1_logit, q2_logit = q1, q2
            critic_loss = self.bce_loss(q1_logit, q).mean() + self.bce_loss(q2_logit, q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def high_actor_loss(self, batch, grad_params, rng=None):
        batch_size, action_dim = batch['high_actor_actions'].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['high_actor_actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        y = x_1 - x_0

        goal_reps = self.network.select('rep_value')(batch['high_actor_goals'])
        pred = self.network.select('high_actor_flow')(
            batch['observations'], goal_reps, x_t, t, params=grad_params
        )
        actor_loss = jnp.mean((pred - y) ** 2)
        return actor_loss, {'actor_loss': actor_loss}

    def low_actor_loss(self, batch, grad_params, rng):
        # Low-level actor's goal is a subgoal observation, not a final goal.
        # The dual paper projects only the high-level goals; low-level subgoal
        # representation is left untouched (it's already in obs space).
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

    # ------- total loss + update -------------------------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, high_actor_rng, low_actor_rng = jax.random.split(rng, 3)

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
        for k, v in high_value_info.items():
            info[f'high_value/{k}'] = v

        high_critic_loss, high_critic_info = self.high_critic_loss(batch, grad_params)
        for k, v in high_critic_info.items():
            info[f'high_critic/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f'rep/{k}'] = v
        rep_contrib = self.config['rep_w'] * rep_loss
        info['rep/rep_contrib'] = rep_contrib

        loss = high_value_loss + high_critic_loss + high_actor_loss + low_actor_loss + rep_contrib
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'high_critic')
        self.target_update(new_network, 'high_value')
        self.target_update(new_network, 'rep_critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        high_seed, low_seed = jax.random.split(seed)

        # Project the final goal through rep_value before high-level rejection sampling.
        goal_reps = self.network.select('rep_value')(goals)

        # High-level: rejection sampling over candidate subgoal *observations*.
        # Critic was trained with goals=psi(final_goal), so we score with the same.
        n_subgoals = jax.random.normal(
            high_seed,
            (self.config['num_samples'], *observations.shape[:-1], self.config['goal_dim']),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        n_goal_reps = jnp.repeat(jnp.expand_dims(goal_reps, 0), self.config['num_samples'], axis=0)

        for i in range(self.config['flow_steps']):
            t = jnp.full((self.config['num_samples'], *observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('high_actor_flow')(n_observations, n_goal_reps, n_subgoals, t)
            n_subgoals = n_subgoals + vels / self.config['flow_steps']

        q = self.network.select('high_critic')(n_observations, goals=n_goal_reps, actions=n_subgoals).min(axis=0)
        subgoals = n_subgoals[jnp.argmax(q)]

        # Low-level: behavioral cloning, conditioned on the *raw* subgoal obs.
        actions = jax.random.normal(low_seed, (*observations.shape[:-1], self.config['action_dim']))
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('low_actor_flow')(observations, subgoals, actions, t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_high_goals = example_batch['high_actor_goals']  # obs-shaped
        ex_value_goals = example_batch['high_value_goals']  # rep_loss reads this
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_high_goals.shape[-1]
        goalrep_dim = config['goalrep_dim']

        # Example projected goal (used to size high-level networks).
        ex_goal_reps = jnp.zeros((ex_high_goals.shape[0], goalrep_dim))

        rep_value_def = DualRepresentationValue(config['rep_type'])(
            hidden_dims=config['rep_hidden_dims'],
            latent_dim=goalrep_dim,
            layer_norm=config['layer_norm'],
        )
        rep_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )

        # High-level networks now consume goalrep_dim-shaped goals.
        high_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        high_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )

        high_actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=goal_dim,  # high-level "actions" are subgoal obs
            layer_norm=config['layer_norm'],
        )
        low_actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )

        network_info = dict(
            rep_value=(rep_value_def, (ex_observations, ex_value_goals)),
            rep_critic=(rep_critic_def, (ex_observations, ex_value_goals, ex_actions)),
            target_rep_critic=(copy.deepcopy(rep_critic_def), (ex_observations, ex_value_goals, ex_actions)),
            high_value=(high_value_def, (ex_observations, ex_goal_reps)),
            target_high_value=(copy.deepcopy(high_value_def), (ex_observations, ex_goal_reps)),
            high_critic=(high_critic_def, (ex_observations, ex_goal_reps, ex_high_goals)),
            target_high_critic=(copy.deepcopy(high_critic_def), (ex_observations, ex_goal_reps, ex_high_goals)),
            high_actor_flow=(high_actor_flow_def, (ex_observations, ex_goal_reps, ex_high_goals, ex_times)),
            low_actor_flow=(low_actor_flow_def, (ex_observations, ex_high_goals, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_high_critic'] = params['modules_high_critic']
        params['modules_target_high_value'] = params['modules_high_value']
        params['modules_target_rep_critic'] = params['modules_rep_critic']

        config['action_dim'] = action_dim
        config['goal_dim'] = goal_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='sharsa_dual',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024, 1024, 1024, 1024),
            value_hidden_dims=(1024, 1024, 1024, 1024),
            rep_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            q_agg='min',
            action_dim=ml_collections.config_dict.placeholder(int),
            goal_dim=ml_collections.config_dict.placeholder(int),
            value_loss_type='bce',
            flow_steps=10,
            num_samples=32,

            # Dual paper rep config (mirrors dual-goal-viscosity defaults).
            rep_type='bilinear',
            goalrep_dim=256,
            rep_expectile=0.9,
            rep_w=1.0,  # weight on the auxiliary rep_loss term

            # Dataset hyperparameters.
            dataset_class='HGCDataset',
            subgoal_steps=25,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=False,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=0.5,
            actor_p_randomgoal=0.5,
            actor_geom_sample=True,
            gc_negative=False,
        )
    )
    return config
