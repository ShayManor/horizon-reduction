import glob
import random
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax
import jax.numpy as jnp
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, HGCDataset
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'Environment name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval.')
flags.DEFINE_integer('num_datasets', None, 'Number of datasets.')
flags.DEFINE_string('save_dir', 'plots/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch (None = auto-detect latest).')
flags.DEFINE_integer('grid_size', 100, 'Grid resolution.')
flags.DEFINE_integer('goal_idx', -1, 'Index of goal state in dataset (-1 = random terminal).')
flags.DEFINE_string('title', None, 'Plot title override (default: agent_name from config).')

# Dummy flags expected by main.py-style configs
flags.DEFINE_integer('offline_steps', 0, '')
flags.DEFINE_integer('log_interval', 10000, '')
flags.DEFINE_integer('eval_interval', 0, '')
flags.DEFINE_integer('save_interval', 0, '')
flags.DEFINE_integer('eval_episodes', 0, '')
flags.DEFINE_float('eval_temperature', 0, '')
flags.DEFINE_float('eval_gaussian', None, '')
flags.DEFINE_integer('video_episodes', 0, '')
flags.DEFINE_integer('video_frame_skip', 3, '')

config_flags.DEFINE_config_file('agent', 'agents/sharsa_geodesic.py', lock_config=False)

# ──────────────────────────────────────────────
# Matplotlib style (from vis.py)
# ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Montserrat", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 14,
    "axes.titlesize": 16,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})

cmap_contour = cm.get_cmap("Greys_r", 20)
contour_colors = [cmap_contour(i) for i in range(cmap_contour.N)][::-1]


# ──────────────────────────────────────────────
# Maze rendering helpers (from vis.py, adapted)
# ──────────────────────────────────────────────
def try_get_maze_info(env):
    """Try to extract maze_map and maze_unit from the environment."""
    env_u = env.unwrapped
    maze_map = getattr(env_u, 'maze_map', None)
    if maze_map is None:
        maze_map = getattr(env_u, '_maze_map', None)
    maze_unit = getattr(env_u, '_maze_unit', None)
    if maze_unit is None:
        maze_unit = getattr(env_u, 'maze_unit', None)
    return maze_map, maze_unit


def plot_maze_on_ax(env, ax):
    """Render maze walls as white rectangles (from vis.py)."""
    maze_map, maze_unit = try_get_maze_info(env)
    if maze_map is None or maze_unit is None:
        print("[warn] Could not find maze_map/maze_unit on env; skipping wall render.")
        return
    maze_map = np.array(maze_map)
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 1:
                wall = np.array([j, i]) * maze_unit - maze_unit * 1.5
                rect = patches.Rectangle(
                    wall, maze_unit, maze_unit,
                    edgecolor=None, facecolor="white", alpha=1, zorder=2,
                )
                ax.add_patch(rect)


def create_meshgrid(env, obs_data, grid_size=100):
    """Create XY meshgrid. Uses maze_map if available, else data bounds."""
    maze_map, maze_unit = try_get_maze_info(env)

    if maze_map is not None and maze_unit is not None:
        maze_map = np.array(maze_map)
        h, w = maze_map.shape
        range_min = -maze_unit * 1.5
        range_max_x = (w - 1) * maze_unit - maze_unit * 1.5 + maze_unit
        range_max_y = (h - 1) * maze_unit - maze_unit * 1.5 + maze_unit
    else:
        pad = 1.0
        range_min_x = obs_data[:, 0].min() - pad
        range_max_x = obs_data[:, 0].max() + pad
        range_min_y = obs_data[:, 1].min() - pad
        range_max_y = obs_data[:, 1].max() + pad
        range_min = min(range_min_x, range_min_y)
        # override per-axis below

    if maze_map is not None and maze_unit is not None:
        x = np.linspace(range_min, range_max_x, grid_size)
        y = np.linspace(range_min, range_max_y, grid_size)
    else:
        x = np.linspace(range_min_x, range_max_x, grid_size)
        y = np.linspace(range_min_y, range_max_y, grid_size)

    X, Y = np.meshgrid(x, y)
    return X, Y


# ──────────────────────────────────────────────
# Core value function query for SHARSA
# ──────────────────────────────────────────────
def query_sharsa_value(agent, grid_points, goals, value_loss_type='bce', chunk_size=5000):
    """Query SHARSA high_value V(s, g) over a grid. Returns numpy array."""

    def value_fn_single(pt, g):
        pt = pt[None, :]
        g = g[None, :]
        v = agent.network.select('high_value')(pt, g)
        if value_loss_type == 'bce':
            v = jax.nn.sigmoid(v)
        return v.squeeze()

    value_fn_batched = jax.jit(jax.vmap(value_fn_single, in_axes=(0, 0)))

    V_chunks = []
    for i in range(0, len(grid_points), chunk_size):
        chunk = value_fn_batched(
            grid_points[i:i + chunk_size],
            goals[i:i + chunk_size],
        )
        V_chunks.append(chunk)
    return np.array(jnp.concatenate(V_chunks))


# ──────────────────────────────────────────────
# Main visualization
# ──────────────────────────────────────────────
def vis_value_function(env, agent, dataset, config):
    """Generate contour plot of V(s, g) over the maze XY plane."""
    obs_all = dataset['observations']
    obs_dim = obs_all.shape[-1]
    obs_mean = obs_all.mean(axis=0)
    grid_size = FLAGS.grid_size

    has_oracle = 'oracle_reps' in dataset

    # ── Pick goal ──
    terminal_locs = np.nonzero(dataset['terminals'] > 0)[0]
    if FLAGS.goal_idx >= 0:
        goal_data_idx = FLAGS.goal_idx
    else:
        goal_data_idx = terminal_locs[np.random.randint(len(terminal_locs))]

    goal_xy = obs_all[goal_data_idx][:2]
    if has_oracle:
        goal_rep = dataset['oracle_reps'][goal_data_idx]
    else:
        goal_rep = obs_all[goal_data_idx]

    print(f"Goal idx={goal_data_idx}, XY=({goal_xy[0]:.2f}, {goal_xy[1]:.2f}), "
          f"goal_rep dim={goal_rep.shape[-1]}")

    # ── Create grid ──
    X, Y = create_meshgrid(env, obs_all, grid_size)
    grid_xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)

    # Pad observation dims > 2 with dataset mean
    if obs_dim > 2:
        padding = np.tile(obs_mean[2:], (grid_xy.shape[0], 1)).astype(np.float32)
        grid_points = np.concatenate([grid_xy, padding], axis=-1)
    else:
        grid_points = grid_xy

    # Tile goal
    goals_all = np.tile(goal_rep, (grid_points.shape[0], 1)).astype(np.float32)

    # ── Query value ──
    print("Querying value function on grid...")
    value_loss_type = config.get('value_loss_type', 'bce')
    V = query_sharsa_value(agent, grid_points, goals_all, value_loss_type)
    V = V.reshape(X.shape)

    print(f"V range: [{V.min():.4f}, {V.max():.4f}]")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 8))

    # Filled contour
    contour = ax.contourf(
        X, Y, V, levels=grid_size, cmap="Blues_r", zorder=0,
    )

    # Soft contour lines
    ax.contour(X, Y, V, levels=80, colors="k", linewidths=0.4, alpha=0.3, zorder=1)

    # Hard contour lines (ruler ticks, from vis.py)
    ax.contour(
        X, Y, V, levels=20,
        colors=contour_colors, linewidths=1.4, alpha=1.0, zorder=1,
        linestyles=[(0, (2, 1))],
    )

    # Maze walls
    plot_maze_on_ax(env, ax)

    # Goal marker
    ax.plot(goal_xy[0], goal_xy[1], 'o',
            color='darkred', markersize=12, markeredgecolor='white',
            markeredgewidth=2, zorder=5, label='Goal')

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(contour, cax=cax, label="Goal-conditioned Value")

    ax.set_aspect('equal')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    title = FLAGS.title if FLAGS.title else config.get('agent_name', 'sharsa')
    ax.set_title(title)
    # ax.legend(loc='upper right')

    # Save
    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = title
    save_path = save_dir / f"{save_name}_value_goal{goal_data_idx}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main(_):
    config = FLAGS.agent

    # ── Load env + dataset (one shard) ──
    if FLAGS.dataset_dir is None:
        datasets = [None]
    else:
        datasets = [f for f in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz'))
                    if '-val.npz' not in f]
    dataset_path = datasets[0]

    env, train_dataset_raw, val_dataset_raw = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path,
    )

    # Keep raw dict for direct access to observations/oracle_reps
    raw_dict = dict(train_dataset_raw)

    # ── Build HGCDataset for agent creation ──
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    train_dataset = HGCDataset(Dataset.create(**train_dataset_raw), config)
    example_batch = train_dataset.sample(1)

    # ── Create + restore agent ──
    # ── Create + restore agent ──
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch, config)

    # Resolve restore path: pick the run with the highest checkpoint epoch.
    restore_epoch = FLAGS.restore_epoch
    candidates = glob.glob(FLAGS.restore_path)
    assert candidates, f'No candidates found for {FLAGS.restore_path}'

    best_path = None
    best_epoch = -1
    for cand in candidates:
        pkl_files = glob.glob(os.path.join(cand, 'params_*.pkl'))
        if not pkl_files:
            continue
        top = max(int(os.path.basename(f).replace('params_', '').replace('.pkl', '')) for f in pkl_files)
        if top > best_epoch:
            best_epoch = top
            best_path = cand

    assert best_path is not None, f'No params_*.pkl found in any candidate: {candidates}'
    print(f"Selected run: {best_path} (epoch {best_epoch})")

    if restore_epoch is None:
        restore_epoch = best_epoch
        print(f"Auto-detected latest checkpoint: epoch {restore_epoch}")

    agent = restore_agent(agent, best_path, restore_epoch)

    # ── Visualize ──
    vis_value_function(env, agent, raw_dict, config)
    print("Done.")


if __name__ == '__main__':
    app.run(main)