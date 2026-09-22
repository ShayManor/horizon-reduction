#!/bin/bash -l
#SBATCH --job-name=spot_geo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/spot_geo_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/spot_geo_%j.err
set -euo pipefail
source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate
export HR_DIR=/scratch/gautschi/manors/hr_sim2real
export HR_EXP_DIR=/scratch/gautschi/manors/exp
export SPOT_DATA=/scratch/gautschi/manors/spot_offline
export WANDB_DIR=/scratch/gautschi/manors/wandb
export WANDB_CACHE_DIR=/scratch/gautschi/manors/wandb-cache
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME
mkdir -p /scratch/gautschi/manors/logs "$HR_EXP_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$HR_DIR"
# Method (d): SHARSA + geodesic HJB. cost_dims=2 keeps c(s, w) on the xy slice.
# Offline training on the Spot lab dataset. eval_interval=0 because evaluating means driving the
# robot and the offline loop has no env; evaluate a checkpoint with --online_steps instead.
python main.py \
  --run_group=spot_geo \
  --seed=${SEED:-1} \
  --env_name=spot-maze-v0 \
  --dataset_dir="$SPOT_DATA" \
  --save_dir="$HR_EXP_DIR" \
  --agent=agents/sharsa_geodesic.py \
  --offline_steps=${STEPS:-10000} \
  --eval_interval=0 \
  --log_interval=500 \
  --save_interval=1000 \
  --agent.gc_negative=True \
  --agent.value_loss_type=squared \
  --agent.subgoal_steps=10 \
  --agent.use_anisotropic=False \
  --agent.w_geo=1.0 \
  --agent.use_fk_loss=False \
  --agent.basic_smoothing=False \
  --agent.cost_dims=2 \
  --agent.kappa=16.7
