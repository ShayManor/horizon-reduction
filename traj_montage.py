"""Stitch the per-(method, task) trajectory panels into one methods x tasks grid.

Reads {method}_task{id}_success*.png from --plot_dir (the layout traj_vis.py
writes), labels each row with the method and each column with the task, and
writes a single composite. Pure image stitching, so it composites whatever the
traj_vis.py run produced without re-querying any checkpoint.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROW_LABELS = {
    'sharsa': 'SHARSA',
    'dual': 'SHARSA+DUAL',
    'phys': 'OURS',
    'geo': 'SHARSA+GEO',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plot_dir', required=True, help='Dir holding the per-panel PNGs.')
    p.add_argument('--methods', default='sharsa,phys', help='Comma-separated method order (rows).')
    p.add_argument('--tasks', default='1,2,3,4,5', help='Comma-separated task ids (columns).')
    p.add_argument('--col_labels', default=None,
                   help='Comma-separated column captions (default: "Task N").')
    p.add_argument('--out', required=True, help='Output composite path.')
    args = p.parse_args()

    methods = args.methods.split(',')
    tasks = args.tasks.split(',')
    if args.col_labels:
        col_labels = args.col_labels.split(',')
        assert len(col_labels) == len(tasks), 'col_labels must match tasks'
    else:
        col_labels = [f'Task {t}' for t in tasks]

    fig, axes = plt.subplots(len(methods), len(tasks),
                             figsize=(3.2 * len(tasks), 2.6 * len(methods)),
                             squeeze=False)
    for r, method in enumerate(methods):
        for c, task in enumerate(tasks):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            matches = sorted(glob.glob(os.path.join(args.plot_dir, f'{method}_task{task}_success*.png')))
            if matches:
                ax.imshow(mpimg.imread(matches[0]))
            else:
                ax.text(0.5, 0.5, f'missing:\n{method} task{task}', ha='center', va='center')
                print(f'[warn] missing {method}_task{task}_success*.png')

            if c == 0:
                ax.set_ylabel(ROW_LABELS.get(method, method.upper()), fontsize=13)
            if r == len(methods) - 1:
                ax.set_xlabel(f'({chr(97 + c)}) {col_labels[c]}', fontsize=13, style='italic')

    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.06, wspace=0.02, hspace=0.02)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved composite to {args.out}')


if __name__ == '__main__':
    main()
