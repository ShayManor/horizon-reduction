import math
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField

_INF = 1e6
_NEIGHBOURS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _xy_to_idx(x, y, xmin, ymin, res, grid_size):
    ix = jnp.clip(jnp.floor((x - xmin) / res).astype(jnp.int32), 0, grid_size - 1)
    iy = jnp.clip(jnp.floor((y - ymin) / res).astype(jnp.int32), 0, grid_size - 1)
    return ix, iy


def _idx_to_xy(ix, iy, xmin, ymin, res):
    return jnp.stack([xmin + (ix + 0.5) * res, ymin + (iy + 0.5) * res], axis=-1)


def _dijkstra_field(occ, src_ix, src_iy, res, iters=80):
    """8-connected shortest-path distance field on `(G, G)` grid with corner-cut prevention."""
    occ = occ.at[src_ix, src_iy].set(False)
    dist = jnp.full(occ.shape, _INF, dtype=jnp.float32).at[src_ix, src_iy].set(0.0)
    o = jnp.pad(occ, 1, constant_values=True)
    o_up, o_dn = o[:-2, 1:-1], o[2:, 1:-1]
    o_lf, o_rt = o[1:-1, :-2], o[1:-1, 2:]
    diag = math.sqrt(2.0) * res

    def _step(_, d):
        p = jnp.pad(d, 1, constant_values=_INF)
        n_up = p[:-2, 1:-1] + res
        n_dn = p[2:, 1:-1] + res
        n_lf = p[1:-1, :-2] + res
        n_rt = p[1:-1, 2:] + res
        n_ul = jnp.where(o_up | o_lf, _INF, p[:-2, :-2] + diag)
        n_ur = jnp.where(o_up | o_rt, _INF, p[:-2, 2:] + diag)
        n_dl = jnp.where(o_dn | o_lf, _INF, p[2:, :-2] + diag)
        n_dr = jnp.where(o_dn | o_rt, _INF, p[2:, 2:] + diag)
        d_min = jnp.minimum(
            jnp.minimum(jnp.minimum(n_up, n_dn), jnp.minimum(n_lf, n_rt)),
            jnp.minimum(jnp.minimum(n_ul, n_ur), jnp.minimum(n_dl, n_dr)),
        )
        d_next = jnp.minimum(d, d_min)
        return jnp.where(occ, _INF, d_next)

    return jax.lax.fori_loop(0, iters, _step, dist)


def _info_gain_grid(visited, occ, radius=4):
    """Fraction of unknown (unvisited and non-occupied) cells within `radius` of each cell."""
    win = 2 * radius + 1
    unknown = ((visited < 0.5) & (~occ)).astype(jnp.float32)
    return jax.lax.reduce_window(unknown, 0.0, jax.lax.add, (win, win), (1, 1), 'SAME') / float(win * win)


class IPPValue(nn.Module):
    """Online grid value V_IPP(s, g) = w_ig * InfoGain(g) - d_grid(s_xy, g_xy)."""

    grid_size: int = 80
    xmin: float = -10.0
    ymin: float = -10.0
    xmax: float = 10.0
    ymax: float = 10.0
    block_threshold: float = 1.0
    dijkstra_iters: int = 80
    ig_radius: int = 4
    ig_weight: float = 2.0

    @nn.compact
    def __call__(self, observations, goals):
        visited = self.param('visited', nn.initializers.zeros, (self.grid_size, self.grid_size), jnp.float32)
        blocked = self.param('blocked', nn.initializers.zeros, (self.grid_size, self.grid_size), jnp.float32)

        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
            goals = goals[None]

        res = (self.xmax - self.xmin) / float(self.grid_size)
        occ = (blocked >= self.block_threshold) & (visited < 0.5)
        ig_grid = _info_gain_grid(visited, occ, self.ig_radius)

        s_xy = observations[:, :2]
        g_xy = goals[:, :2]
        s_ix, s_iy = _xy_to_idx(s_xy[:, 0], s_xy[:, 1], self.xmin, self.ymin, res, self.grid_size)
        g_ix, g_iy = _xy_to_idx(g_xy[:, 0], g_xy[:, 1], self.xmin, self.ymin, res, self.grid_size)
        euclid = jnp.linalg.norm(s_xy - g_xy, axis=-1)

        def _from_single_goal(_):
            dist_field = _dijkstra_field(occ, g_ix[0], g_iy[0], res, self.dijkstra_iters)
            return dist_field[s_ix, s_iy]

        def _from_single_obs(_):
            dist_field = _dijkstra_field(occ, s_ix[0], s_iy[0], res, self.dijkstra_iters)
            return dist_field[g_ix, g_iy]

        def _general_batch(_):
            def _pair_dist(six, siy, gix, giy):
                df = _dijkstra_field(occ, gix, giy, res, self.dijkstra_iters)
                return df[six, siy]

            return jax.vmap(_pair_dist)(s_ix, s_iy, g_ix, g_iy)

        same_goal = jnp.all(g_ix == g_ix[:1]) & jnp.all(g_iy == g_iy[:1])
        same_obs = jnp.all(s_ix == s_ix[:1]) & jnp.all(s_iy == s_iy[:1])
        d_grid = jax.lax.cond(
            same_goal,
            _from_single_goal,
            lambda _: jax.lax.cond(same_obs, _from_single_obs, _general_batch, operand=None),
            operand=None,
        )

        # Sub-cell Euclidean tie-breaker keeps value strictly monotone within a grid cell.
        d_total = d_grid + 0.05 * euclid
        ig = ig_grid[g_ix, g_iy]
        v = self.ig_weight * ig - d_total
        return v[0] if unbatched else v


class IPPAgent(flax.struct.PyTreeNode):
    """Classical Grid IPP baseline: online occupancy/visit grid + InfoGain/Cost frontier + Dijkstra subgoal routing."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

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
        """Total loss for gradient-updated modules (low_actor_flow). Grid params are updated in `update`."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, low_actor_rng = jax.random.split(rng, 2)

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        if self.config['freeze_low_actor']:
            low_actor_loss = jax.lax.stop_gradient(low_actor_loss)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        v_s = self.network.select('high_value')(batch['observations'][:16], batch['high_value_goals'][:16])
        info['high_value/v_mean'] = v_s.mean()
        info['total_loss'] = low_actor_loss
        return low_actor_loss, info

    def _update_grid_params(self, grid_params, batch):
        """Update `visited` and `blocked` grid counts from transitions in `batch`."""
        G = self.config['grid_size']
        xmin, ymin = self.config['grid_xmin'], self.config['grid_ymin']
        res = (self.config['grid_xmax'] - xmin) / float(G)

        obs = batch['observations']
        next_obs = batch['next_observations']
        actions = batch['actions']

        visited = grid_params['visited']
        blocked = grid_params['blocked']

        # Mark visited cells at both `obs` and `next_obs`.
        s_ix, s_iy = _xy_to_idx(obs[:, 0], obs[:, 1], xmin, ymin, res, G)
        n_ix, n_iy = _xy_to_idx(next_obs[:, 0], next_obs[:, 1], xmin, ymin, res, G)
        visited = visited.at[s_ix, s_iy].set(1.0)
        visited = visited.at[n_ix, n_iy].set(1.0)

        # Detect blocked contacts: commanded planar speed > 0.2 m/s while measured speed < 0.05 m/s or zero step.
        vx_cmd = actions[:, 0] * 0.6
        vy_cmd = actions[:, 1] * 0.4
        cmd_speed = jnp.hypot(vx_cmd, vy_cmd)
        meas_speed = jnp.hypot(next_obs[:, 4], next_obs[:, 5])
        disp = jnp.hypot(next_obs[:, 0] - obs[:, 0], next_obs[:, 1] - obs[:, 1])
        is_blk = (cmd_speed > 0.2) & ((meas_speed < 0.05) | (disp < 1e-3))

        # World-frame unit direction of the blocked command.
        cos_yaw, sin_yaw = obs[:, 2], obs[:, 3]
        wx = cos_yaw * vx_cmd - sin_yaw * vy_cmd
        wy = sin_yaw * vx_cmd + cos_yaw * vy_cmd
        wnorm = jnp.hypot(wx, wy) + 1e-6
        ux, uy = wx / wnorm, wy / wnorm

        # Obstacle cell sits one grid cell ahead of the robot in the blocked command direction.
        obs_x = obs[:, 0] + res * ux
        obs_y = obs[:, 1] + res * uy
        b_ix, b_iy = _xy_to_idx(obs_x, obs_y, xmin, ymin, res, G)
        blocked = blocked.at[b_ix, b_iy].add(is_blk.astype(jnp.float32))

        return {'visited': visited, 'blocked': blocked}

    @jax.jit
    def update(self, batch):
        """Update the online occupancy/visit grid and low-level actor."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_params = dict(new_network.params)
        new_params['modules_high_value'] = self._update_grid_params(new_params['modules_high_value'], batch)
        new_network = new_network.replace(params=new_params)

        info['ipp/visited_cells'] = jnp.sum(new_params['modules_high_value']['visited'] >= 0.5)
        info['ipp/blocked_cells'] = jnp.sum(
            (new_params['modules_high_value']['blocked'] >= self.config['block_threshold'])
            & (new_params['modules_high_value']['visited'] < 0.5)
        )
        return self.replace(network=new_network, rng=new_rng), info

    def _single_subgoal(self, observation, goal):
        """Route a lookahead subgoal along the 8-connected Dijkstra shortest path on the discovered grid."""
        G = self.config['grid_size']
        xmin, ymin = self.config['grid_xmin'], self.config['grid_ymin']
        res = (self.config['grid_xmax'] - xmin) / float(G)

        hv_params = self.network.params['modules_high_value']
        visited = hv_params['visited']
        blocked = hv_params['blocked']
        occ = (blocked >= self.config['block_threshold']) & (visited < 0.5)

        s_xy = observation[:2]
        g_xy = goal[:2]
        s_ix, s_iy = _xy_to_idx(s_xy[0], s_xy[1], xmin, ymin, res, G)
        g_ix, g_iy = _xy_to_idx(g_xy[0], g_xy[1], xmin, ymin, res, G)

        dist = _dijkstra_field(occ, g_ix, g_iy, res, self.config['dijkstra_iters'])
        # Radial tie-breaker aligns open-space paths with the true straight line to goal.
        xs = xmin + (jnp.arange(G) + 0.5) * res
        ys = ymin + (jnp.arange(G) + 0.5) * res
        radial = jnp.hypot(xs[:, None] - g_xy[0], ys[None, :] - g_xy[1])
        field = jnp.where(occ, _INF, dist + 1e-2 * radial).at[g_ix, g_iy].set(0.0)

        def _descent_step(_, cur):
            cix, ciy = cur
            best_ix, best_iy = cix, ciy
            best_val = field[cix, ciy]
            for dx, dy in _NEIGHBOURS:
                nix = jnp.clip(cix + dx, 0, G - 1)
                niy = jnp.clip(ciy + dy, 0, G - 1)
                corner_cut = (dx != 0) & (dy != 0) & (occ[nix, ciy] | occ[cix, niy])
                val = jnp.where(corner_cut, _INF, field[nix, niy])
                better = val < best_val
                best_ix = jnp.where(better, nix, best_ix)
                best_iy = jnp.where(better, niy, best_iy)
                best_val = jnp.where(better, val, best_val)
            return (best_ix, best_iy)

        cur_ix, cur_iy = jax.lax.fori_loop(0, self.config['lookahead_cells'], _descent_step, (s_ix, s_iy))
        subgoal_cell_xy = _idx_to_xy(cur_ix, cur_iy, xmin, ymin, res)

        # Snap to exact goal coordinates when the lookahead reaches the goal cell.
        reached_goal_cell = (cur_ix == g_ix) & (cur_iy == g_iy)
        subgoal_xy = jnp.where(reached_goal_cell, g_xy, subgoal_cell_xy)

        # Fallback to straight-line step if the start cell is disconnected on the grid.
        diff = g_xy - s_xy
        d_goal = jnp.linalg.norm(diff)
        fallback = jnp.where(
            d_goal <= self.config['subgoal_step_dist'],
            g_xy,
            s_xy + self.config['subgoal_step_dist'] * diff / (d_goal + 1e-6),
        )
        disconnected = dist[s_ix, s_iy] >= _INF * 0.5
        return jnp.where(disconnected, fallback, subgoal_xy)

    @jax.jit
    def select_subgoal(self, observations, goals, seed=None):
        """Return the Dijkstra lookahead subgoal for (observations, goals)."""
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
        """Route a subgoal via Dijkstra on the discovered grid and execute via the low-level flow policy."""
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

        high_value_def = IPPValue(
            grid_size=config['grid_size'],
            xmin=config['grid_xmin'],
            ymin=config['grid_ymin'],
            xmax=config['grid_xmax'],
            ymax=config['grid_ymax'],
            block_threshold=config['block_threshold'],
            dijkstra_iters=config['dijkstra_iters'],
            ig_radius=config['ig_radius'],
            ig_weight=config['ig_weight'],
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
            agent_name='ipp',
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
            # Classical Grid IPP parameters.
            grid_size=80,
            grid_xmin=-10.0,
            grid_ymin=-10.0,
            grid_xmax=10.0,
            grid_ymax=10.0,
            block_threshold=1.0,
            dijkstra_iters=80,
            lookahead_cells=4,
            subgoal_step_dist=1.0,
            ig_radius=4,
            ig_weight=2.0,
        )
    )
    return config
