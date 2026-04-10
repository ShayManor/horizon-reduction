#!/bin/bash -l
#SBATCH --job-name=vis_sharsa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/vis_sharsa_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/vis_sharsa_%j.err
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

RESTORE_PATH="exp/horizon-reduction/geo_aniso_p45_100m_dev/*"
RESTORE_EPOCH=1000000

# Dataset + env (antmaze with oraclerep for XY contour plots)
ENV_NAME="antmaze-medium-navigate-oraclerep-v0"
DATASET_DIR="$BASE/antmaze-medium-navigate-100m-v0"

# Which agent config was used for training
AGENT_CONFIG="agents/sharsa_geodesic.py"

# ─── Baseline SHARSA (uncomment to compare) ───
# RESTORE_PATH="exp/horizon-reduction/baseline_sharsa_antmaze/*"
# AGENT_CONFIG="agents/sharsa.py"

# ─── RUN ───
python vis_sharsa.py \
  --env_name="$ENV_NAME" \
  --dataset_dir="$DATASET_DIR" \
  --agent="$AGENT_CONFIG" \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --restore_path="$RESTORE_PATH" \
  --restore_epoch=$RESTORE_EPOCH \
  --grid_size=100 \
  --goal_idx=-1 \
  --save_dir=plots/
