"""Stitch the four per-method value plots into a single 2x2 composite.

Reads {method}_{tag}_value_goal{goal_idx}.png for method in the default order
(sharsa, dual, phys, geo) from --plot_dir and writes a 2x2 grid to --out. Pure
image stitching (matplotlib imread/imshow), so it composites whatever the vis.py
run produced without re-querying any checkpoint.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plot_dir', required=True, help='Dir holding the per-method PNGs.')
    p.add_argument('--tag', required=True, help='Env tag, e.g. antmed / humgiant.')
    p.add_argument('--goal_idx', type=int, default=42)
    p.add_argument('--methods', default='sharsa,dual,phys,geo',
                   help='Comma-separated method order (row-major over the 2x2).')
    p.add_argument('--out', required=True, help='Output composite path.')
    args = p.parse_args()

    methods = args.methods.split(',')
    assert len(methods) == 4, f'Expected 4 methods, got {methods}'

    fig, axes = plt.subplots(2, 2, figsize=(20, 17))
    for ax, method in zip(axes.ravel(), methods):
        ax.axis('off')
        # Glob the goal suffix: with --goal_xy the resolved dataset index isn't
        # known to the caller, so match whatever index vis.py wrote.
        pat = os.path.join(args.plot_dir, f'{method}_{args.tag}_value_goal*.png')
        hits = sorted(glob.glob(pat))
        if hits:
            ax.imshow(mpimg.imread(hits[0]))
        else:
            ax.text(0.5, 0.5, f'missing:\n{method}_{args.tag}', ha='center', va='center')
            print(f'[warn] missing {pat}')

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved composite to {args.out}')


if __name__ == '__main__':
    main()
