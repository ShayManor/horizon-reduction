#!/bin/bash -l
#SBATCH --job-name=sharsa_pointmazelarge_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/sharsa_pointmazelarge_1b_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/sharsa_pointmazelarge_1b_%j.err
set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export BASE=/scratch/gautschi/manors/ogbench
export HR_EXP_DIR=/scratch/gautschi/manors/exp
export WANDB_DIR=/scratch/gautschi/manors/wandb
export WANDB_CACHE_DIR=/scratch/gautschi/manors/wandb-cache
export HR_DIR=$HOME/horizon-reduction
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

mkdir -p /scratch/gautschi/manors/logs "$HR_EXP_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$HR_DIR"

# Method (a): baseline SHARSA on pointmaze-large 1b.
python main.py \
  --run_group=sharsa_pointmazelarge_1b \
  --seed=1 \
  --env_name=pointmaze-large-navigate-v0 \
  --dataset_dir="$BASE/pointmaze-large-navigate-1b-v0" \
  --dataset_replace_interval=1000 \
  --agent=agents/sharsa.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.discount=0.99 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True \
  --offline_steps=1000000 \
  --eval_interval=50000 \
  --save_interval=500000 \
  --log_interval=50000 \
  --video_episodes=0
