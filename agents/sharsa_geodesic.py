import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCValue, GCMetric


class SHARSAGeodesicAgent(flax.struct.PyTreeNode):
    """SHARSA + Geodesic HJB regularization agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        loss = -(log_pred * target + log_not_pred * (1 - target))
        return loss

    @staticmethod
    def _masked_mean(x, mask):
        return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-6)

    def high_value_loss(self, batch, grad_params):
        """Compute the high-level SARSA value loss."""
        q1, q2 = self.network.select('target_high_critic')(
            batch['observations'], goals=batch['high_value_goals'], actions=batch['high_value_actions']
        )
        if self.config['value_loss_type'] == 'bce':
            q1, q2 = jax.nn.sigmoid(q1), jax.nn.sigmoid(q2)

        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        elif self.config['q_agg'] == 'mean':
            q = (q1 + q2) / 2

        v = self.network.select('high_value')(batch['observations'], batch['high_value_goals'], params=grad_params)
        if self.config['value_loss_type'] == 'bce':
            v_logit = v
            v = jax.nn.sigmoid(v_logit)

        if self.config['value_loss_type'] == 'squared':
            value_loss = ((v - q) ** 2).mean()
        elif self.config['value_loss_type'] == 'bce':
            value_loss = (self.bce_loss(v_logit, q)).mean()

        goal_type = batch['high_value_goal_type']
        cur_mask = (goal_type == 0).astype(jnp.float32)
        traj_mask = (goal_type == 1).astype(jnp.float32)
        rand_mask = (goal_type == 2).astype(jnp.float32)

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_std': v.std(),
            'v_max': v.max(),
            'v_min': v.min(),
            'v_mean_cur': self._masked_mean(v, cur_mask),
            'v_mean_traj': self._masked_mean(v, traj_mask),
            'v_mean_rand': self._masked_mean(v, rand_mask),
            'q_mean_cur': self._masked_mean(q, cur_mask),
            'q_mean_traj': self._masked_mean(q, traj_mask),
            'q_mean_rand': self._masked_mean(q, rand_mask),
            'v_minus_q_mean': (v - q).mean(),
            'frac_cur': cur_mask.mean(),
            'frac_traj': traj_mask.mean(),
            'frac_rand': rand_mask.mean(),
        }

    def high_critic_loss(self, batch, grad_params):
        """Compute the high-level SARSA critic loss."""
        next_v = self.network.select('high_value')(batch['high_value_next_observations'], batch['high_value_goals'])
        if self.config['value_loss_type'] == 'bce':
            next_v = jax.nn.sigmoid(next_v)
        discount_k = self.config['discount'] ** batch['high_value_subgoal_steps']
        bootstrap = discount_k * batch['high_value_masks'] * next_v
        q = batch['high_value_rewards'] + bootstrap

        q1, q2 = self.network.select('high_critic')(
            batch['observations'], batch['high_value_goals'], batch['high_value_actions'], params=grad_params
        )

        if self.config['value_loss_type'] == 'squared':
            critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        elif self.config['value_loss_type'] == 'bce':
            q1_logit, q2_logit = q1, q2
            critic_loss = self.bce_loss(q1_logit, q).mean() + self.bce_loss(q2_logit, q).mean()

        goal_type = batch['high_value_goal_type']
        cur_mask = (goal_type == 0).astype(jnp.float32)
        traj_mask = (goal_type == 1).astype(jnp.float32)
        rand_mask = (goal_type == 2).astype(jnp.float32)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_std': q.std(),
            'q_max': q.max(),
            'q_min': q.min(),
            'reward_mean': batch['high_value_rewards'].mean(),
            'bootstrap_mean': bootstrap.mean(),
            'next_v_mean': next_v.mean(),
            'success_frac': (1.0 - batch['high_value_masks']).mean(),
            'discount_k_mean': discount_k.mean(),
            'q_target_mean_traj': self._masked_mean(q, traj_mask),
            'q_target_mean_rand': self._masked_mean(q, rand_mask),
            'q_target_mean_cur': self._masked_mean(q, cur_mask),
        }

    def high_actor_loss(self, batch, grad_params, rng=None):
        """Compute the high-level flow BC actor loss."""
        batch_size, action_dim = batch['high_actor_actions'].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['high_actor_actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        y = x_1 - x_0

        pred = self.network.select('high_actor_flow')(
            batch['observations'], batch['high_actor_goals'], x_t, t, params=grad_params
        )

        actor_loss = jnp.mean((pred - y) ** 2)

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info

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

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info

    def geodesic_hjb_loss(self, batch, grad_params):
        """Geodesic HJB / basic-smoothing regularization with unified diagnostics."""
        obs = batch['observations']  # (B, D)
        goals = batch['high_value_goals']  # (B, G)
        w = batch['high_value_next_observations']  # (B, obs_dim)

        # --- Value at s and w (logits if bce) ---
        v_s_raw = self.network.select('high_value')(obs, goals, params=grad_params)
        v_w_raw = self.network.select('high_value')(w, goals, params=grad_params)

        if self.config['value_loss_type'] == 'bce':
            v_s = jax.nn.sigmoid(v_s_raw)
            v_w = jax.nn.sigmoid(v_w_raw)
        else:
            v_s = v_s_raw
            v_w = v_w_raw

        # --- Cost c(s, w) (always computed for diagnostics) ---
        delta = w - obs  # (B, obs_dim)
        delta_norm = jnp.linalg.norm(delta, axis=-1)

        if self.config['use_anisotropic']:
            U, scale = self.network.select('metric')(obs, params=grad_params)
            Ut_delta = jnp.einsum('...dr,...d->...r', U, delta)
            sq_dist = scale * jnp.sum(delta ** 2, axis=-1) + jnp.sum(Ut_delta ** 2, axis=-1)
            cost = jnp.sqrt(sq_dist + 1e-8)
        else:
            cost = delta_norm * self.config['kappa']

        # --- HJB residual: V(w) - V(s) + c. Violation iff < 0. Always computed. ---
        hjb_residual = v_w - v_s + cost
        v_gap = v_w - v_s  # signed V difference (key for diagnosing smoothing asymmetry)

        # --- Goal-type masks for stratified diagnostics ---
        goal_type = batch['high_value_goal_type']
        cur_mask = (goal_type == 0).astype(jnp.float32)
        traj_mask = (goal_type == 1).astype(jnp.float32)
        rand_mask = (goal_type == 2).astype(jnp.float32)

        # --- Violation diagnostics ---
        violation = (hjb_residual < 0.0).astype(jnp.float32)
        violation_frac = violation.mean()
        violation_mag = jnp.abs(jnp.minimum(0.0, hjb_residual))  # 0 where satisfied
        violation_mag_mean = violation_mag.mean()

        # --- Tightness diagnostic (independent of whether we use tight loss) ---
        tau = self.config['tightness_threshold']
        tight_mask = ((hjb_residual >= 0.0) & (hjb_residual < tau)).astype(jnp.float32)
        tight_frac = tight_mask.mean()

        # Diagnostics dict — populated in both branches.
        diag = {
            # HJB diagnostics
            'hjb_residual_mean': hjb_residual.mean(),
            'hjb_residual_std': hjb_residual.std(),
            'hjb_residual_min': hjb_residual.min(),
            'hjb_residual_max': hjb_residual.max(),
            'violation_frac': violation_frac,
            'violation_mag_mean': violation_mag_mean,
            'violation_frac_cur': self._masked_mean(violation, cur_mask),
            'violation_frac_traj': self._masked_mean(violation, traj_mask),
            'violation_frac_rand': self._masked_mean(violation, rand_mask),
            'tight_frac': tight_frac,
            # V gap diagnostics — tests the sampling-asymmetry hypothesis directly
            'v_gap_mean': v_gap.mean(),
            'v_gap_mean_cur': self._masked_mean(v_gap, cur_mask),
            'v_gap_mean_traj': self._masked_mean(v_gap, traj_mask),
            'v_gap_mean_rand': self._masked_mean(v_gap, rand_mask),
            'v_gap_abs_mean': jnp.abs(v_gap).mean(),
            'v_gap_abs_mean_logit': jnp.abs(v_w_raw - v_s_raw).mean(),
            # Cost diagnostics
            'cost_mean': cost.mean(),
            'cost_std': cost.std(),
            'cost_max': cost.max(),
            'delta_norm_mean': delta_norm.mean(),
            # V at endpoints
            'v_s_mean': v_s.mean(),
            'v_w_mean': v_w.mean(),
        }

        # --- Loss selection: basic smoothing OR full HJB combo ---
        if self.config['basic_smoothing']:
            # Basic smoothing operates on raw outputs (logits when value_loss_type='bce').
            loss_smooth = jnp.square(v_w_raw - v_s_raw).mean()
            geo_loss = loss_smooth
            diag.update({
                'geo_loss': geo_loss,
                'loss_smooth': loss_smooth,
                'loss_hjb': 0.0,
                'loss_tight': 0.0,
                'loss_multi': 0.0,
                'loss_metric_reg': 0.0,
                'contrib_hjb': 0.0,
                'contrib_tight': 0.0,
                'contrib_multi': 0.0,
                'contrib_metric': 0.0,
                'contrib_smooth': loss_smooth,
            })
            return geo_loss, diag

        # --- Full HJB-regularizer path ---
        loss_hjb = jnp.square(jnp.minimum(0.0, hjb_residual)).mean()
        loss_tight = (tight_mask * jnp.square(hjb_residual)).sum() / (tight_mask.sum() + 1e-6)

        # Multi-step residual (currently reuses w; contribution logged separately).
        steps = batch['high_value_subgoal_steps']
        v_next = v_w  # same point as w — see note above
        if self.config['use_anisotropic']:
            multi_delta = delta
            Ut_md = jnp.einsum('...dr,...d->...r', U, multi_delta)
            sq_md = scale * jnp.sum(multi_delta ** 2, axis=-1) + jnp.sum(Ut_md ** 2, axis=-1)
            multi_cost = jnp.sqrt(sq_md + 1e-8)
        else:
            multi_cost = self.config['kappa'] * steps
        multi_residual = v_next - v_s + multi_cost
        loss_multi = jnp.square(jnp.minimum(0.0, multi_residual)).mean()

        if self.config['use_anisotropic']:
            loss_metric_reg = jnp.mean(jnp.sum(U ** 2, axis=(-2, -1)))
        else:
            loss_metric_reg = 0.0

        w_hjb = self.config['w_hjb']
        w_tight = self.config['w_tight']
        w_multi = self.config['w_multi']
        w_metric = self.config['w_metric']

        contrib_hjb = w_hjb * loss_hjb
        contrib_tight = w_tight * loss_tight
        contrib_multi = w_multi * loss_multi
        contrib_metric = w_metric * loss_metric_reg
        geo_loss = contrib_hjb + contrib_tight + contrib_multi + contrib_metric

        diag.update({
            'geo_loss': geo_loss,
            'loss_hjb': loss_hjb,
            'loss_tight': loss_tight,
            'loss_multi': loss_multi,
            'loss_metric_reg': loss_metric_reg,
            'loss_smooth': 0.0,
            'contrib_hjb': contrib_hjb,
            'contrib_tight': contrib_tight,
            'contrib_multi': contrib_multi,
            'contrib_metric': contrib_metric,
            'contrib_smooth': 0.0,
            'multi_residual_mean': multi_residual.mean(),
            'multi_cost_mean': multi_cost.mean(),
        })
        return geo_loss, diag

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss = SHARSA losses + geodesic HJB."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, high_value_rng, high_critic_rng, high_actor_rng, low_actor_rng = jax.random.split(rng, 5)

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

        # NEW: geodesic HJB regularization
        geo_loss, geo_info = self.geodesic_hjb_loss(batch, grad_params)
        for k, v in geo_info.items():
            info[f'geodesic/{k}'] = v

        w_geo = self.config['w_geo']
        reg_contrib = w_geo * geo_loss
        bce_main = high_value_loss + high_critic_loss
        info['geodesic/reg_over_bce_ratio'] = reg_contrib / (bce_main + 1e-8)
        info['geodesic/reg_contrib'] = reg_contrib

        loss = high_value_loss + high_critic_loss + high_actor_loss + low_actor_loss + reg_contrib
        info['total_loss'] = loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'high_critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
            self,
            observations,
            goals=None,
            seed=None,
            temperature=1.0,
    ):
        """Sample actions from the actor."""
        high_seed, low_seed = jax.random.split(seed)

        # High-level: rejection sampling.
        orig_observations = observations
        n_subgoals = jax.random.normal(
            high_seed,
            (
                self.config['num_samples'],
                *observations.shape[:-1],
                self.config['goal_dim'],
            ),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        n_goals = jnp.repeat(jnp.expand_dims(goals, 0), self.config['num_samples'], axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), self.config['num_samples'], axis=0)

        for i in range(self.config['flow_steps']):
            t = jnp.full((self.config['num_samples'], *observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('high_actor_flow')(n_observations, n_goals, n_subgoals, t)
            n_subgoals = n_subgoals + vels / self.config['flow_steps']

        q = self.network.select('high_critic')(n_orig_observations, goals=n_goals, actions=n_subgoals).min(axis=0)
        subgoals = n_subgoals[jnp.argmax(q)]

        # Low-level: behavioral cloning.
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
        ex_goals = example_batch['high_actor_goals']
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]
        state_dim = ex_observations.shape[-1]

        # Same networks as SHARSA
        high_value_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                                 num_ensembles=1)
        high_critic_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                                  num_ensembles=2)
        high_actor_flow_def = ActorVectorField(hidden_dims=config['actor_hidden_dims'], action_dim=goal_dim,
                                               layer_norm=config['layer_norm'])
        low_actor_flow_def = ActorVectorField(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim,
                                              layer_norm=config['layer_norm'])

        # NEW: metric tensor network
        metric_def = GCMetric(
            hidden_dims=config['metric_hidden_dims'],
            state_dim=state_dim,
            rank=config['metric_rank'],
            layer_norm=config['layer_norm'],
        )

        network_info = dict(
            high_value=(high_value_def, (ex_observations, ex_goals)),
            high_critic=(high_critic_def, (ex_observations, ex_goals, ex_goals)),
            target_high_critic=(copy.deepcopy(high_critic_def), (ex_observations, ex_goals, ex_goals)),
            high_actor_flow=(high_actor_flow_def, (ex_observations, ex_goals, ex_goals, ex_times)),
            low_actor_flow=(low_actor_flow_def, (ex_observations, ex_goals, ex_actions, ex_times)),
            metric=(metric_def, (ex_observations,)),  # NEW
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_high_critic'] = params['modules_high_critic']

        config['action_dim'] = action_dim
        config['goal_dim'] = goal_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='sharsa_geodesic',
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
            value_loss_type='bce',
            flow_steps=10,
            num_samples=32,
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

            # Geodesic HJB hyperparams
            basic_smoothing=False,
            use_anisotropic=False,
            metric_hidden_dims=(512, 512),
            metric_rank=8,
            kappa=0.01,  # isotropic fallback cost scale
            tightness_threshold=0.2,
            w_geo=0.5,  # overall weight of geodesic loss
            w_hjb=1.0,
            w_tight=0.5,
            w_multi=0.5,
            w_metric=0.01,
        )
    )
    return config
