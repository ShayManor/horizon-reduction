#!/bin/bash -l
#SBATCH --job-name=dual_humanoidgiant_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/dual_humanoidgiant_1b_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/dual_humanoidgiant_1b_%j.err
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

# Method (b): SHARSA + dual goal paper loss on humanoidmaze-giant 1b.
apply_resume_args "dual_humanoidgiant_1b"

python main.py \
  --run_group=dual_humanoidgiant_1b \
  --seed=1 \
  --env_name=humanoidmaze-giant-navigate-oraclerep-v0 \
  --dataset_dir="$BASE/humanoidmaze-giant-navigate-1b-v0" \
  --dataset_replace_interval=4000 \
  --agent=agents/sharsa_dual.py \
  --agent.q_agg=mean \
  --agent.subgoal_steps=50 \
  --agent.actor_p_trajgoal=1.0 \
  --agent.actor_p_randomgoal=0.0 \
  --agent.actor_geom_sample=False \
  --agent.rep_type=bilinear \
  --agent.goalrep_dim=256 \
  --agent.rep_expectile=0.9 \
  --agent.rep_w=1.0 \
  --offline_steps=3000000 \
  --eval_interval=250000 \
  --save_interval=500000 \
  --log_interval=50000 \
  --video_episodes=0 \
  $RESUME_ARGS
