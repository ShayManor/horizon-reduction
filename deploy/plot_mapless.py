"""Figures and the mapping-time table for mapless runs (spec §5).

    python deploy/plot_mapless.py --runs=sharsa=<run dir>,dual=<run dir>,... --out=plots/mapless_<date>/

Each run directory needs `scores.csv` (deploy/score_checkpoints.py), `train.csv` and `warmup.csv`
(CsvLogger fixes its header on the first row, so the warmup summaries live in their own file and are
joined to `train.csv` on `progress/env_steps`). `value_snapshots.png` also needs the agent, `steps.csv`,
`flags.json` and the ground-truth grid, and is scoring-only like the scorer.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

# This script lives in deploy/, so Python puts that directory on the path rather than the repo
# root, and the repo's own packages would not import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SR_MIN, COVERAGE_MIN = 0.8, 0.9  # mapping thresholds
ARMS = ['sharsa', 'dual', 'phys', 'geo']
COLORS = {'sharsa': '#4c78a8', 'dual': '#f58518', 'phys': '#54a24b', 'geo': '#b279a2'}
SPARE = ['#e45756', '#72b7b2', '#9d755d', '#bab0ac']  # arms outside ARMS, in order of appearance
SNAPSHOT_STEPS = (2000, 5000)  # plus the final checkpoint
HEADINGS = np.arange(8) * math.pi / 4
MAX_CELLS = 6000  # grid cells V is evaluated on per snapshot


def read_csv(path):
    """Columns as float arrays (blank and non-numeric cells are nan), or None when the file is missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    def num(v):
        try:
            return float({'True': 1, 'False': 0}.get(v, v))
        except (TypeError, ValueError):
            return math.nan

    return {k: np.array([num(r[k]) for r in rows]) for k in rows[0]}


def load_progress(run_dir):
    """`warmup.csv` and `train.csv` joined on `progress/env_steps`, as columns sorted by env step."""
    parts = [p for p in (read_csv(os.path.join(run_dir, n)) for n in ('warmup.csv', 'train.csv')) if p]
    keys = [k for k in parts[0] if all(k in p for p in parts)] if parts else []
    cols = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if not cols:
        return {}
    order = np.argsort(cols['progress/env_steps'], kind='stable')
    return {k: v[order] for k, v in cols.items()}


def mapping_time(run_dir):
    """First checkpoint step with fixed SR >= 0.8 and coverage >= 0.9, or None when no step has both.

    Coverage at a step is the last progress row at or before it. Wall time and metres are interpolated
    between the progress rows around the step.
    """
    scores, progress = read_csv(os.path.join(run_dir, 'scores.csv')), load_progress(run_dir)
    if not scores or not progress:
        return None
    steps = progress['progress/env_steps']
    for step, sr in zip(scores['step'], scores['fixed_SR']):
        i = np.searchsorted(steps, step, side='right') - 1
        if sr >= SR_MIN and i >= 0 and progress['explore/coverage'][i] >= COVERAGE_MIN:
            return dict(
                step=int(step),
                wall_time=float(np.interp(step, steps, progress['progress/wall_time'])),
                metres=float(np.interp(step, steps, progress['progress/metres'])),
            )
    return None


def parse_runs(spec):
    runs = {}
    for item in spec.split(','):
        arm, _, path = item.partition('=')
        assert path, f'--runs wants arm=run_dir pairs, got {item!r}'
        runs[arm] = path
    ordered = [a for a in ARMS if a in runs] + [a for a in runs if a not in ARMS]
    return {a: runs[a] for a in ordered}


def arm_colors(arms):
    spare = iter(SPARE)
    return {a: COLORS.get(a) or next(spare) for a in arms}


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)


def _at_scores(scores, progress, key):
    """`progress[key]` interpolated at each checkpoint step."""
    return np.interp(scores['step'], progress['progress/env_steps'], progress[key])


def line_figure(path, data, colors, xlabel, ylabel, x_of, y_of, markers=None):
    """One line per arm. `x_of` and `y_of` map a run's data to arrays; `markers` maps arm to an (x, y) point."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for arm, d in data.items():
        x, y = x_of(d), y_of(d)
        if x is None or y is None:
            continue
        ax.plot(x, y, marker='o', ms=3, lw=1.5, color=colors[arm], label=arm)
        if markers and markers.get(arm):
            ax.plot(*markers[arm], marker='*', ms=12, color=colors[arm], mec='black', mew=0.8, ls='none')
    _style(ax, xlabel, ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def timing_figure(path, data, colors):
    keys = [('time/update_ms', 'update (ms)'), ('time/act_ms', 'act (ms)'), ('time/overrun_frac', 'overrun fraction')]
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 3.2))
    for ax, (key, label) in zip(axes, keys):
        for arm, d in data.items():
            p = d['progress']
            if key in p:
                ax.plot(p['progress/env_steps'], p[key], marker='o', ms=3, lw=1.5, color=colors[arm], label=arm)
        _style(ax, 'env steps', label)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _run_frame(run_flags, start_mark, identity):
    from deploy.eval_pairs import load_start_mark
    from envs.spot_mapless import RunFrame
    from envs.spot_maze import SIM_HOSTNAME

    if identity:
        return RunFrame((0.0, 0.0, 0.0))
    if run_flags.get('robot_hostname') == SIM_HOSTNAME and run_flags.get('start_pose') is not None:
        return RunFrame([float(v) for v in run_flags['start_pose']])
    return load_start_mark(start_mark)


def _nearest(available, target):
    return min(available, key=lambda s: abs(s - target))


def snapshot_steps(run_dir, wanted=SNAPSHOT_STEPS):
    """Available checkpoint steps nearest each of `wanted` and the final one, without repeats."""
    names = glob.glob(os.path.join(run_dir, 'params_*.pkl'))
    available = sorted(int(os.path.basename(p)[len('params_') : -len('.pkl')]) for p in names)
    picks = [_nearest(available, w) for w in wanted] + [available[-1]]
    return sorted(set(picks))


def grid_values(agent, client, frame, goal_seed, stride):
    """V(s, g) averaged over HEADINGS on every `stride`-th grid cell, nan on occupied cells. Returns (xs, ys, V)."""
    from utils.value_query import pair_values

    rows, cols = np.arange(0, client.free.shape[0], stride), np.arange(0, client.free.shape[1], stride)
    xs, ys = client.lo[0] + (cols + 0.5) * client.res, client.lo[1] + (rows + 0.5) * client.res
    free = client.free[np.ix_(rows, cols)]
    rr, cc = np.nonzero(free)
    cells_run = np.array([frame.to_run(xs[c], ys[r], 0.0)[:2] for r, c in zip(rr, cc)])
    n, h = len(cells_run), len(HEADINGS)
    obs = np.zeros((n * h, 7), dtype=np.float32)
    obs[:, :2] = np.repeat(cells_run, h, axis=0)
    obs[:, 2], obs[:, 3] = np.tile(np.cos(HEADINGS), n), np.tile(np.sin(HEADINGS), n)
    goal = np.repeat(np.array([frame.to_run(*goal_seed, 0.0)[:2]], dtype=np.float32), n * h, axis=0)
    values = np.full(free.shape, np.nan)
    values[rr, cc] = pair_values(agent, obs, goal).reshape(n, h).mean(1)
    return xs, ys, values


def value_snapshots(path, runs, colors, start_mark, identity, occupancy_path, goal_index, wanted=SNAPSHOT_STEPS):
    """Rows are arms, columns checkpoints: V toward one goal over the free cells, with the walk so far."""
    from deploy.eval_pairs import value_goals
    from deploy.score_checkpoints import load_agent, scoring_env
    from deploy.sim_client import SimSpotClient

    picks, loaded = {}, {}
    for arm, run_dir in runs.items():
        with open(os.path.join(run_dir, 'flags.json')) as f:
            run_flags = json.load(f)
        picks[arm] = snapshot_steps(run_dir, wanted)
        loaded[arm] = (
            run_flags,
            _run_frame(run_flags, start_mark, identity),
            read_csv(os.path.join(run_dir, 'steps.csv')),
        )
    cols = max(len(p) for p in picks.values())
    first = next(iter(loaded.values()))[0]
    client = SimSpotClient(
        occupancy_path,
        {},
        goal_tol=first['goal_tol'],
        seed=0,
        control_hz=first.get('control_hz', 10.0),
        vel_limits=tuple(float(v) for v in first['vel_limits']),
        reverse_limit=first.get('reverse_limit'),
    )
    goal = value_goals(client, k=8, seed=0)[goal_index]
    stride = max(1, int(math.ceil(math.sqrt(client.free.size / MAX_CELLS))))
    extent = (client.lo[0], client.lo[0] + client.free.shape[1] * client.res,
              client.lo[1], client.lo[1] + client.free.shape[0] * client.res)  # fmt: skip

    fig, axes = plt.subplots(len(runs), cols, figsize=(3.6 * cols, 3.0 * len(runs)), squeeze=False)
    for r, (arm, run_dir) in enumerate(runs.items()):
        run_flags, frame, steps = loaded[arm]
        env = scoring_env(client, frame, run_flags)
        for c in range(cols):
            ax = axes[r, c]
            if c >= len(picks[arm]):
                ax.axis('off')
                continue
            step = picks[arm][c]
            agent = load_agent(run_dir, step, run_flags['agent'], env)
            xs, ys, v = grid_values(agent, client, frame, goal, stride)
            ax.imshow(
                client.free, origin='lower', extent=extent, cmap='gray', vmin=0.0, vmax=1.3, interpolation='nearest'
            )
            cs = ax.contourf(xs, ys, np.ma.masked_invalid(v), levels=12, cmap='viridis', alpha=0.85)
            if steps and 'x' in steps:
                keep = (steps['step'] <= step) & (steps['jump'] == 0)
                walk = np.array([frame.from_run(x, y, 0.0)[:2] for x, y in zip(steps['x'][keep], steps['y'][keep])])
                if len(walk):
                    ax.plot(walk[:, 0], walk[:, 1], color='white', lw=0.6, alpha=0.9)
            ax.plot(*goal, marker='*', ms=11, color=colors[arm], mec='black', mew=0.8, ls='none')
            ax.set_aspect('equal')
            ax.set_xlabel(f'{step / 1000:g}k steps' if r == len(runs) - 1 else '')
            ax.set_ylabel(arm if c == 0 else '')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[['top', 'right']].set_visible(False)
            fig.colorbar(cs, ax=ax, shrink=0.7, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    from envs.spot_mapless import _REPO, SIM_OCCUPANCY_PATH

    p = argparse.ArgumentParser()
    p.add_argument(
        '--runs', required=True, help='arm=run_dir,... in any order; colours follow sharsa, dual, phys, geo.'
    )
    p.add_argument('--out', required=True)
    p.add_argument('--start_mark', default=os.path.join(_REPO, 'spot_data', 'start_mark.json'))
    p.add_argument('--identity_frame', action='store_true')
    p.add_argument('--occupancy_path', default=SIM_OCCUPANCY_PATH)
    p.add_argument('--goal_index', type=int, default=0, help='Which of the 8 value goals the snapshots use.')
    p.add_argument('--no_snapshots', action='store_true', help='Skip value_snapshots.png (it loads every agent).')
    args = p.parse_args()

    runs = parse_runs(args.runs)
    colors = arm_colors(runs)
    os.makedirs(args.out, exist_ok=True)
    data = {a: dict(scores=read_csv(os.path.join(d, 'scores.csv')), progress=load_progress(d)) for a, d in runs.items()}
    marks = {a: mapping_time(d) for a, d in runs.items()}

    def out(name):
        return os.path.join(args.out, f'{name}.png')

    def scored(key):
        return lambda d: d['scores'][key] if d['scores'] else None

    def steps_of(d):
        return d['scores']['step'] if d['scores'] else None

    def progress_x(key):
        return lambda d: _at_scores(d['scores'], d['progress'], key) if d['scores'] and d['progress'] else None

    sr = scored('fixed_SR')
    line_figure(out('sr_vs_steps'), data, colors, 'env steps', 'fixed-set SR', steps_of, sr)
    wall = progress_x('progress/wall_time')

    def hours(d):
        t = wall(d)
        return None if t is None else t / 3600

    line_figure(out('sr_vs_time'), data, colors, 'wall-clock (h)', 'fixed-set SR', hours, sr)
    line_figure(
        out('sr_vs_metres'), data, colors, 'distance walked (m)', 'fixed-set SR', progress_x('progress/metres'), sr
    )
    line_figure(
        out('length_ratio'), data, colors, 'env steps', 'path length / geodesic', steps_of, scored('fixed_length_ratio')
    )
    line_figure(
        out('coverage'), data, colors, 'env steps', 'coverage',
        lambda d: d['progress'].get('progress/env_steps'), lambda d: d['progress'].get('explore/coverage'),
        markers={a: (m['step'], _coverage_at(runs[a], m['step'])) if m else None for a, m in marks.items()},
    )  # fmt: skip
    line_figure(
        out('value_spearman'), data, colors, 'env steps', 'Spearman(V, -geodesic)', steps_of, scored('value_spearman')
    )
    timing_figure(out('timing'), data, colors)
    if not args.no_snapshots:
        value_snapshots(
            out('value_snapshots'),
            runs,
            colors,
            args.start_mark,
            args.identity_frame,
            args.occupancy_path,
            args.goal_index,
        )

    print(f'mapping time (fixed SR >= {SR_MIN}, coverage >= {COVERAGE_MIN})')
    print(f'{"arm":<10}{"step":>10}{"wall (s)":>12}{"metres":>10}')
    for arm, m in marks.items():
        print(
            f'{arm:<10}' + (f'{m["step"]:>10d}{m["wall_time"]:>12.0f}{m["metres"]:>10.0f}' if m else f'{"absent":>10}')
        )
    print(f'wrote figures to {args.out}')


def _coverage_at(run_dir, step):
    p = load_progress(run_dir)
    return float(p['explore/coverage'][np.searchsorted(p['progress/env_steps'], step, side='right') - 1])


if __name__ == '__main__':
    main()
