import glob
import random
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, LogNorm

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
flags.DEFINE_string('goal_xy', None, 'Target goal position as "x,y". Overrides --goal_idx: '
                    'picks the dataset observation whose xy is closest to it.')
flags.DEFINE_bool('nn_pose', False, 'Fill non-xy dims per grid cell from the nearest '
                  'real dataset observation (on-manifold) instead of a single pose.')
flags.DEFINE_integer('nn_k', 1, 'k for --nn_pose: average the poses of the k nearest '
                     'real states per cell (k>1 smooths out per-cell velocity jitter).')
flags.DEFINE_string('title', None, 'Plot title override (default: agent_name from config).')
flags.DEFINE_string('highlight_xy', None, 'Region to ring as "x,y". Set it to compare the '
                    'same patch of maze across methods; leave unset to ring each '
                    'run\'s own auto-detected traps instead.')

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

MAZE_GRAY = "#808080"
# Soft teal-blue ramp (navy = far from goal, teal = near). Kept off pure black so
# the white iso-lines stay legible in the far field.
CMAP_VALUE = LinearSegmentedColormap.from_list('tealblue', ['#153067', '#69BBB4'])
BASIN_COLOR = "#FF8C00"


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
    """Render maze walls as grey rectangles, then a faint cell grid over them."""
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
                    edgecolor=None, facecolor=MAZE_GRAY, alpha=1, zorder=2,
                )
                ax.add_patch(rect)

    h, w = maze_map.shape
    for j in range(w + 1):
        ax.axvline(j * maze_unit - maze_unit * 1.5, color="white",
                   lw=0.6, alpha=0.13, zorder=3)
    for i in range(h + 1):
        ax.axhline(i * maze_unit - maze_unit * 1.5, color="white",
                   lw=0.6, alpha=0.13, zorder=3)


def free_space_mask(env, X, Y):
    """Boolean grid: True where the (x, y) sample lands in a non-wall maze cell."""
    maze_map, maze_unit = try_get_maze_info(env)
    if maze_map is None or maze_unit is None:
        return np.ones(X.shape, bool)
    maze_map = np.array(maze_map)
    h, w = maze_map.shape
    jj = np.clip(((X + maze_unit * 1.5) / maze_unit).astype(int), 0, w - 1)
    ii = np.clip(((Y + maze_unit * 1.5) / maze_unit).astype(int), 0, h - 1)
    return maze_map[ii, jj] == 0


def find_closed_extrema(V, X, Y, free, maze_unit, goal_xy, iso_levels, min_bands=1):
    """Locate the closed rings in V — free-space points that are an extremum
    against every point on a ring one maze-cell out.

    A geodesic value field has exactly one extremum, the goal. Anything else is a
    trap: a local *max* is a false attractor a greedy planner climbs into, a local
    *min* is a basin it stalls in. Both draw closed contour rings, so both count.

    Depth is measured in *iso-levels crossed*, not in raw value: V is a discounted
    return spanning orders of magnitude, so any fixed fraction of the global range
    would only ever fire near the goal. `min_bands` is therefore literally "how
    many closed contour rings this feature draws".

    Returns a list of (x, y, kind) with kind in {'max', 'min'}.
    """
    from scipy import ndimage as ndi

    dx = (X.max() - X.min()) / (X.shape[1] - 1)
    r = max(2, int(round(maze_unit / dx)))

    # Drop a border of width r: the filters' edge padding makes the grid corners
    # look like extrema.
    valid = free.copy()
    valid[:r, :] = valid[-r:, :] = valid[:, :r] = valid[:, -r:] = False
    # The goal is the one legitimate extremum; its surroundings can ring.
    valid &= np.hypot(X - goal_xy[0], Y - goal_xy[1]) > 1.5 * maze_unit

    ring = np.ones((2 * r + 1, 2 * r + 1), bool)
    ring[1:-1, 1:-1] = False
    lv = np.asarray(iso_levels)

    def bands_between(a, b):
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        return ((lv[None, None, :] > lo[..., None])
                & (lv[None, None, :] < hi[..., None])).sum(-1)

    found = []
    # Search the *unmasked* field: V is continuous across walls, and blanking them
    # out would make every wall rim look like an extremum. Free space is required
    # of the centre only, via `valid`.
    for kind, peak, edge in (('min', ndi.minimum_filter, ndi.minimum_filter),
                             ('max', ndi.maximum_filter, ndi.maximum_filter)):
        is_ext = (V == peak(V, size=2 * r + 1, mode='nearest')) & valid
        ring_v = edge(V, footprint=ring, mode='nearest')
        keep = is_ext & (bands_between(V, ring_v) >= min_bands)
        lbl, n = ndi.label(keep)
        for k in range(1, n + 1):
            iy, ix = np.where(lbl == k)
            found.append((X[iy, ix].mean(), Y[iy, ix].mean(), kind))
    return found


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
    grid_size = FLAGS.grid_size

    has_oracle = 'oracle_reps' in dataset

    # ── Pick goal ──
    terminal_locs = np.nonzero(dataset['terminals'] > 0)[0]
    if FLAGS.goal_xy is not None:
        target_xy = np.array([float(v) for v in FLAGS.goal_xy.split(',')], dtype=np.float32)
        goal_data_idx = int(np.argmin(np.linalg.norm(obs_all[:, :2] - target_xy, axis=-1)))
    elif FLAGS.goal_idx >= 0:
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

    # Pad observation dims > 2 with a real dataset pose. Two modes:
    #  - single pose (default, HIQL/OGBench d4rl_ant.py convention): reuse one
    #    recorded pose at every cell. On-manifold globally but the goal cell gets
    #    the wrong body pose, so V can dip there (full-obs-goal envs like antmaze).
    #  - nearest-neighbor (--nn_pose): each grid cell borrows the proprioceptive
    #    dims of the closest real dataset observation at that (x,y). Every cell —
    #    including the goal cell — gets a locally valid pose, so V peaks on-goal.
    if obs_dim > 2:
        if FLAGS.nn_pose:
            from scipy.spatial import cKDTree
            tree = cKDTree(obs_all[:, :2])
            k = max(1, FLAGS.nn_k)
            _, nn_idx = tree.query(grid_xy, k=k, workers=-1)
            if k == 1:
                padding = obs_all[nn_idx, 2:].astype(np.float32)
            else:
                # Average the poses of the k nearest real states → smooth, still
                # locally-valid pose field (kills the per-cell velocity jitter a
                # single nearest neighbor produces).
                padding = obs_all[nn_idx, 2:].mean(axis=1).astype(np.float32)
        else:
            padding = np.tile(obs_all[0, 2:], (grid_xy.shape[0], 1)).astype(np.float32)
        grid_points = np.concatenate([grid_xy, padding], axis=-1)
    else:
        grid_points = grid_xy

    # Tile goal
    goals_all = np.tile(goal_rep, (grid_points.shape[0], 1)).astype(np.float32)

    # Dual-goal agents feed high_value a learned goal embedding psi(g), not the
    # raw goal. Encode the goals through rep_value so the value net input matches.
    if 'modules_rep_value' in agent.network.params:
        goals_all = np.asarray(agent.network.select('rep_value')(goals_all)).astype(np.float32)

    # ── Query value ──
    print("Querying value function on grid...")
    value_loss_type = config.get('value_loss_type', 'bce')
    V = query_sharsa_value(agent, grid_points, goals_all, value_loss_type)
    V = V.reshape(X.shape)

    print(f"V range: [{V.min():.4f}, {V.max():.4f}]")

    # Sanity check: V should peak on the goal cell. A large offset means the grid
    # states are off-manifold at the goal (wrong body pose), not a marker bug.
    peak_iy, peak_ix = np.unravel_index(np.argmax(V), V.shape)
    peak_xy = (X[peak_iy, peak_ix], Y[peak_iy, peak_ix])
    print(f"Goal marker at ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}); "
          f"V peaks at ({peak_xy[0]:.2f}, {peak_xy[1]:.2f}); "
          f"offset {np.hypot(peak_xy[0] - goal_xy[0], peak_xy[1] - goal_xy[1]):.2f}")

    # ── Plot ── (plasma fill + black dashed iso-lines over grey walls. Lines
    # trace a lightly smoothed copy of the field so they read as clean curves.)
    # Size the canvas to the maze so the equal-aspect axes fill it (a square
    # figure leaves a wide maze short and its labels oversized after tight bbox).
    fig, ax = plt.subplots(figsize=(8 * (X.max() - X.min()) / (Y.max() - Y.min()), 8))
    ax.set_facecolor(MAZE_GRAY)

    # nn_pose borrows a real pose per cell, which leaves a little per-cell jitter
    # in V. Lightly smooth the fill (and more for the iso-lines) so the map reads
    # cleanly; single-pose renders keep the raw fill they always had.
    try:
        from scipy.ndimage import gaussian_filter
        V_fill = gaussian_filter(V, sigma=1.5, mode="nearest") if FLAGS.nn_pose else V
        V_iso = gaussian_filter(V, sigma=3.0 if FLAGS.nn_pose else 2.0, mode="nearest")
    except Exception:
        V_fill, V_iso = V, V

    # V is a discounted return (~gamma^steps-to-goal), so equal *ratios* in V are
    # equal steps of distance: log spacing puts a band every fixed number of steps
    # all the way across the maze, where linear spacing crams every level into the
    # goal's neighborhood and leaves the far field a featureless floor. Quantile
    # spacing was worse still — it forced a fixed line count into the flat region,
    # carving the noise floor into closed blobs. Fall back to linear if a
    # non-sigmoid value head puts V at or below zero.
    log_scale = V_fill.min() > 0 and V_iso.min() > 0
    spacing = np.geomspace if log_scale else np.linspace

    # Filled contour
    ax.contourf(
        X, Y, V_fill, levels=spacing(V_fill.min(), V_fill.max(), grid_size),
        norm=LogNorm(V_fill.min(), V_fill.max()) if log_scale else None,
        cmap=CMAP_VALUE, zorder=0,
    )

    # Iso-lines, white on the teal-blue ramp. Every 5th is drawn heavier so the
    # bands stay readable where the minor lines crowd together.
    iso_levels = spacing(V_iso.min(), V_iso.max(), 65)[1:-1]
    ax.contour(
        X, Y, V_iso, levels=iso_levels,
        colors="white", linewidths=0.6, alpha=0.55, zorder=1,
        linestyles=[(0, (1, 2))],
    )
    ax.contour(
        X, Y, V_iso, levels=iso_levels[::5],
        colors="white", linewidths=1.4, alpha=0.95, zorder=1,
        linestyles=[(0, (4, 3))],
    )

    # Maze walls
    plot_maze_on_ax(env, ax)

    # Ring the closed basins — local minima of V, where a greedy planner stalls.
    _, maze_unit = try_get_maze_info(env)
    if maze_unit is not None:
        basins = find_closed_extrema(
            V_iso, X, Y, free_space_mask(env, X, Y), maze_unit, goal_xy, iso_levels)
        print(f"Closed extrema (planner traps): {len(basins)} -> "
              + ", ".join(f"{k} at ({x:.1f}, {y:.1f})" for x, y, k in basins))

        # With --highlight_xy every panel rings the same patch, so the methods can
        # be read against each other; otherwise ring whatever each run detected.
        if FLAGS.highlight_xy is not None:
            hx, hy = (float(v) for v in FLAGS.highlight_xy.split(','))
            rings = [(hx, hy)]
        else:
            rings = [(bx, by) for bx, by, _ in basins]
        for bx, by in rings:
            ax.add_patch(patches.Circle(
                (bx, by), radius=1.4 * maze_unit, fill=False,
                edgecolor=BASIN_COLOR, linewidth=2.4, zorder=6))

    # Goal marker
    ax.plot(goal_xy[0], goal_xy[1], 'o',
            color='red', markersize=12, markeredgecolor='white',
            markeredgewidth=1.5, zorder=7)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = FLAGS.title if FLAGS.title else config.get('agent_name', 'sharsa')
    ax.set_title(title)

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
        epochs = [int(os.path.basename(f).replace('params_', '').replace('.pkl', '')) for f in pkl_files]
        # When a specific epoch is pinned, only runs that actually saved that
        # checkpoint are eligible: a single run_group can hold fragmented resume
        # dirs that skip early epochs. Among eligible runs pick the most complete.
        if restore_epoch is not None and restore_epoch not in epochs:
            continue
        top = max(epochs)
        if top > best_epoch:
            best_epoch = top
            best_path = cand

    if restore_epoch is not None:
        assert best_path is not None, (
            f'No run under {FLAGS.restore_path} contains params_{restore_epoch}.pkl')
    else:
        assert best_path is not None, f'No params_*.pkl found in any candidate: {candidates}'
    print(f"Selected run: {best_path} (max epoch {best_epoch})")

    if restore_epoch is None:
        restore_epoch = best_epoch
        print(f"Auto-detected latest checkpoint: epoch {restore_epoch}")

    agent = restore_agent(agent, best_path, restore_epoch)

    # ── Visualize ──
    vis_value_function(env, agent, raw_dict, config)
    print("Done.")


if __name__ == '__main__':
    app.run(main)