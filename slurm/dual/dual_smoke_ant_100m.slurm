#!/bin/bash -l
#SBATCH --job-name=dual_smoke_ant_100m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/dual_smoke_ant_100m_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/dual_smoke_ant_100m_%j.err
set -euo pipefail

# Smoke test for agents/sharsa_dual.py: short run to verify shapes/JIT/training
# work end-to-end before launching the full matrix. Not for evaluation.

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

# First run the unit tests; abort if they fail.
python -m pytest tests/test_sharsa_dual.py -x -v

# Then a 5k-step training smoke run on antmaze-medium-100m.
python main.py \
  --run_group=dual_smoke_ant_100m \
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
  --offline_steps=5000 \
  --eval_interval=5000 \
  --save_interval=5000 \
  --log_interval=500 \
  --video_episodes=0 \
  --eval_episodes=10
