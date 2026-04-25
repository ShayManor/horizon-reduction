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
mkdir -p "$HR_DIR/plots"
cd "$HR_DIR"

ENV_NAME="antmaze-medium-navigate-v0"
DATASET_DIR="$BASE/antmaze-medium-navigate-1b-v0"

GOAL_IDX=42
COMMON="--env_name=$ENV_NAME --dataset_dir=$DATASET_DIR --grid_size=100 --goal_idx=$GOAL_IDX --save_dir=plots/"

# Baseline SHARSA
echo "=== Visualizing: Baseline SHARSA ==="
python vis.py \
  --agent=agents/sharsa.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --restore_path="exp/horizon-reduction/baseline_sharsa_ant_1b/*" \
  $COMMON

# Basic Smoothing
echo "=== Visualizing: Basic Smoothing ==="
python vis.py \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.basic_smoothing=True \
  --agent.w_geo=0.5 \
  --restore_path="exp/horizon-reduction/basic_smoothing_ant_1b/sd000_*" \
  --title=basic_smoothing \
  $COMMON

# Geodesic (isotropic)
echo "=== Visualizing: Geodesic Isotropic ==="
python vis.py \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.use_anisotropic=False \
  --agent.w_geo=1.0 \
  --restore_path="exp/horizon-reduction/geo_aniso_ant_1b_dev/sd000_*" \
  --title=sharsa_geodesic \
  $COMMON

echo "=== All visualizations complete ==="
ls -la plots/*.png

# Commit and push generated plots.
echo "=== Committing plots to git ==="
cd "$HR_DIR"
git add plots/*.png
if git diff --cached --quiet; then
  echo "No plot changes to commit."
else
  git -c user.name="shaymanor" -c user.email="shay.manor@gmail.com" commit -m "vis: update value contour plots ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  git pull --rebase --autostash origin "$(git rev-parse --abbrev-ref HEAD)" || true
  git push origin "$(git rev-parse --abbrev-ref HEAD)"
fi