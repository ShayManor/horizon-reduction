#!/bin/bash -l
#SBATCH --job-name=dual_ant_100m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/dual_ant_100m_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/dual_ant_100m_%j.err
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

# Method (b): SHARSA + dual goal paper loss. rep_loss (paper P2) added to SHARSA total_loss;
# goals projected through rep_value(g). No phys/FK.
python main.py \
  --run_group=dual_ant_100m \
  --seed=1 \
  --env_name=antmaze-medium-navigate-v0 \
  --dataset_dir="$BASE/antmaze-medium-navigate-100m-v0" \
  --dataset_replace_interval=1000 \
  --agent=agents/sharsa_dual.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True \
  --agent.rep_type=bilinear \
  --agent.goalrep_dim=256 \
  --agent.rep_expectile=0.9 \
  --agent.rep_w=1.0 \
  --offline_steps=5000000 \
  --eval_interval=250000 \
  --save_interval=500000 \
  --log_interval=50000 \
  --video_episodes=0
