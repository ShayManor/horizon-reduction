#!/bin/bash -l
#SBATCH --job-name=vis_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/vis_all_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/vis_all_%j.err
set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export BASE=/scratch/gautschi/manors/ogbench
export HR_DIR=$HOME/horizon-reduction
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
unset DISPLAY
unset JAX_PLATFORM_NAME

mkdir -p /scratch/gautschi/manors/logs
cd "$HR_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$HR_DIR/plots/$STAMP"
mkdir -p "$OUT_DIR"
echo "=== Output dir: $OUT_DIR ==="

ENV_NAME="antmaze-medium-navigate-v0"
DATASET_DIR="$BASE/antmaze-medium-navigate-1b-v0"

GOAL_IDX=42
COMMON="--env_name=$ENV_NAME --dataset_dir=$DATASET_DIR --grid_size=100 --goal_idx=$GOAL_IDX --save_dir=$OUT_DIR/"

# Resolve the most recent training run directory under a run_group (by mtime).
latest_run() {
  local group="$1"
  ls -1dt "exp/horizon-reduction/$group"/sd*/ 2>/dev/null | head -n 1 || true
}

run_vis() {
  local label="$1"; shift
  local group="$1"; shift
  local path
  path=$(latest_run "$group")
  if [ -z "$path" ]; then
    echo "=== Skipping: $label (no run found under exp/horizon-reduction/$group) ==="
    return 0
  fi
  path="${path%/}"
  echo "=== Visualizing: $label  ($path) ==="
  python vis.py "$@" --restore_path="$path" $COMMON
}

# Baseline SHARSA
run_vis "Baseline SHARSA" \
  "baseline_sharsa_ant_1b" \
  --agent=agents/sharsa.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --title=baseline_sharsa

# Basic Smoothing (skipped if not trained yet)
run_vis "Basic Smoothing" \
  "basic_smoothing_ant_1b" \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.basic_smoothing=True \
  --agent.w_geo=0.5 \
  --title=basic_smoothing

# Geodesic (isotropic)
run_vis "Geodesic Isotropic" \
  "geo_aniso_ant_1b_dev" \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.use_anisotropic=False \
  --agent.w_geo=1.0 \
  --title=sharsa_geodesic

echo "=== All visualizations complete ==="
ls -la "$OUT_DIR"/*.png 2>/dev/null || echo "(no plots produced)"

# Commit and push generated plots.
echo "=== Committing plots to git ==="
cd "$HR_DIR"
git add "plots/$STAMP"/*.png 2>/dev/null || true
if git diff --cached --quiet; then
  echo "No plot changes to commit."
else
  git -c user.name="shaymanor" -c user.email="shay.manor@gmail.com" commit -m "vis: value contour plots $STAMP"
  git pull --rebase --autostash origin "$(git rev-parse --abbrev-ref HEAD)" || true
  git push origin "$(git rev-parse --abbrev-ref HEAD)"
fi