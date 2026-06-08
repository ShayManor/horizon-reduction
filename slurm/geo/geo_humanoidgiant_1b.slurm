#!/bin/bash -l
#SBATCH --job-name=geo_humanoidgiant_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/geo_humanoidgiant_1b_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/geo_humanoidgiant_1b_%j.err
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

source "$HR_DIR/slurm/_resume.sh"

# Method (d): SHARSA + geodesic HJB on humanoidmaze-giant 1b.
apply_resume_args "geo_humanoidgiant_1b"

python main.py \
  --run_group=geo_humanoidgiant_1b \
  --seed=1 \
  --env_name=humanoidmaze-giant-navigate-v0 \
  --dataset_dir="$BASE/humanoidmaze-giant-navigate-1b-v0" \
  --dataset_replace_interval=1000 \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True \
  --agent.use_anisotropic=False \
  --agent.w_geo=1.0 \
  --agent.tightness_threshold=0.1 \
  --agent.metric_rank=8 \
  --offline_steps=5000000 \
  --eval_interval=250000 \
  --save_interval=500000 \
  --log_interval=50000 \
  --video_episodes=0 \
  $RESUME_ARGS
