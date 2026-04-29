#!/bin/bash -l
#SBATCH --job-name=fk_p45_100m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/fk_p45_100m_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/fk_p45_100m_%j.err
set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export BASE=/scratch/gautschi/manors/ogbench
export HR_DIR=$HOME/horizon-reduction
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

mkdir -p /scratch/gautschi/manors/logs
cd "$HR_DIR"

python main.py \
  --run_group=fk_p45_100m \
  --seed=0 \
  --env_name=puzzle-4x5-play-oraclerep-v0 \
  --dataset_dir="$BASE/puzzle-4x5-play-100m-v0" \
  --dataset_replace_interval=1000 \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True \
  --agent.use_anisotropic=False \
  --agent.w_geo=1.0 \
  --agent.tightness_threshold=0.2 \
  --agent.metric_rank=8 \
  --agent.use_fk_loss=True \
  --agent.viscous_scale=0.01 \
  --agent.num_walks=10 \
  --offline_steps=1000000 \
  --eval_interval=250000 \
  --save_interval=500000 \
  --log_interval=50000 \
  --video_episodes=0
