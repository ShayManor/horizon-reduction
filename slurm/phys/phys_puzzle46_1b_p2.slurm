#!/bin/bash -l
#SBATCH --job-name=phys_puzzle46_1b_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/phys_puzzle46_1b_p2_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/phys_puzzle46_1b_p2_%j.err
set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export BASE=/scratch/gautschi/manors/ogbench
export HR_EXP_DIR=/scratch/gautschi/manors/exp
export WANDB_DIR=/scratch/gautschi/manors/wandb
export WANDB_CACHE_DIR=/scratch/gautschi/manors/wandb-cache
export JAX_COMPILATION_CACHE_DIR=/scratch/gautschi/manors/xla-cache
export HR_DIR=$HOME/horizon-reduction
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

mkdir -p /scratch/gautschi/manors/logs "$HR_EXP_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$JAX_COMPILATION_CACHE_DIR"
cd "$HR_DIR"

source "$HR_DIR/slurm/_resume.sh"

# Method (c): SHARSA + phys (FK fixed-point) on puzzle-4x6 1b.
export RESUME=true  # part 2: resume from the part-1 checkpoint (~3M) and train to 5M
apply_resume_args "phys_puzzle46_1b"

python main.py \
  --run_group=phys_puzzle46_1b \
  --seed=${SEED:-1} \
  --env_name=puzzle-4x6-play-oraclerep-v0 \
  --dataset_dir="$BASE/puzzle-4x6-play-1b-v0" \
  --dataset_replace_interval=1000 \
  --agent=agents/sharsa.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True \
  --agent.w_fk=1.0 \
  --agent.viscous_scale=0.01 \
  --agent.num_walks=10 \
  --offline_steps=5000000 \
  --eval_interval=250000 \
  --save_interval=500000 \
  --log_interval=50000 \
  --video_episodes=0 \
  $RESUME_ARGS
