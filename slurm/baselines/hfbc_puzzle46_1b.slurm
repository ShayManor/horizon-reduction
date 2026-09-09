#!/bin/bash -l
#SBATCH --job-name=hfbc_puzzle46_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/hfbc_puzzle46_1b_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/hfbc_puzzle46_1b_%j.err
set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export BASE=/scratch/gautschi/manors/ogbench
export HR_EXP_DIR=/scratch/gautschi/manors/exp
export WANDB_DIR=/scratch/gautschi/manors/wandb
export WANDB_CACHE_DIR=/scratch/gautschi/manors/wandb-cache
export HR_DIR=$HOME/horizon-reduction
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

mkdir -p /scratch/gautschi/manors/logs "$HR_EXP_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$HR_DIR"

source "$HR_DIR/slurm/_resume.sh"

# Paper baseline: hierarchical flow BC on puzzle-4x6-play-oraclerep-v0 1b.
# Flags from README.md "Reproducing the main results".
# 5M steps does not fit the 12h smallgpu cap on every cell; split it as
#   STEPS=3000000 sbatch slurm/baselines/hfbc_puzzle46_1b.slurm
#   RESUME=true sbatch slurm/baselines/hfbc_puzzle46_1b.slurm
apply_resume_args "hfbc_puzzle46_1b"

python main.py \
  --run_group=hfbc_puzzle46_1b \
  --seed=${SEED:-1} \
  --env_name=puzzle-4x6-play-oraclerep-v0 \
  --dataset_dir="$BASE/puzzle-4x6-play-1b-v0" \
  --dataset_replace_interval=1000 \
  --agent=agents/hgcfbc.py \
  --agent.subgoal_steps=25 \
  --offline_steps=${STEPS:-5000000} \
  --eval_interval=250000 \
  --save_interval=500000 \
  --log_interval=50000 \
  --video_episodes=0 \
  $RESUME_ARGS
