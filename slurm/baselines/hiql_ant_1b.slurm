#!/bin/bash -l
#SBATCH --job-name=hiql_ant_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/hiql_ant_1b_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/hiql_ant_1b_%j.err
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

# Paper baseline: HIQL on antmaze-medium-navigate-v0 1b.
# Flags from README.md "Reproducing the main results".
# antmaze-medium is not in the paper, so these reuse the puzzle-4x6 column,
# which is also this agent's get_config() default. alpha is untuned here.
# 5M steps does not fit the 12h smallgpu cap on every cell; split it as
#   STEPS=3000000 sbatch slurm/baselines/hiql_ant_1b.slurm
#   RESUME=true sbatch slurm/baselines/hiql_ant_1b.slurm
apply_resume_args "hiql_ant_1b"

python main.py \
  --run_group=hiql_ant_1b \
  --seed=${SEED:-1} \
  --env_name=antmaze-medium-navigate-v0 \
  --dataset_dir="$BASE/antmaze-medium-navigate-1b-v0" \
  --dataset_replace_interval=1000 \
  --agent=agents/hiql.py \
  --agent.expectile=0.7 \
  --agent.low_alpha=3 \
  --agent.high_alpha=3 \
  --agent.actor_subgoal_steps=25 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True \
  --offline_steps=${STEPS:-5000000} \
  --eval_interval=250000 \
  --save_interval=500000 \
  --log_interval=50000 \
  --video_episodes=0 \
  $RESUME_ARGS
