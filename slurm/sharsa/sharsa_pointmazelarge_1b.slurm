#!/bin/bash -l
#SBATCH --job-name=sharsa_pointmazelarge_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
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

source "$HR_DIR/slurm/_resume.sh"

# Bisection experiment: baseline SHARSA on pointmaze-large using the STANDARD
# OGBench dataset (no --dataset_dir -> OGBench loads its built-in dataset).
# If this works, the custom 1b dataset is the culprit; if it still fails, the
# bug is in our SHARSA pointmaze config/eval, not the data.
apply_resume_args "sharsa_pointmazelarge_std"

# Materialize the standard OGBench dataset into the cache before training so the
# run fails fast with a clear message if the compute node has no internet
# (instead of stalling mid-init). No-op if already cached.
python -c "import ogbench; ogbench.make_env_and_datasets('pointmaze-large-navigate-v0', compact_dataset=True)"

python main.py \
  --run_group=sharsa_pointmazelarge_std \
  --seed=${SEED:-1} \
  --env_name=pointmaze-large-navigate-v0 \
  --agent=agents/sharsa.py \
  --agent.q_agg=mean \
  --agent.subgoal_steps=50 \
  --agent.actor_p_trajgoal=1.0 \
  --agent.actor_p_randomgoal=0.0 \
  --agent.actor_geom_sample=False \
  --offline_steps=1000000 \
  --eval_interval=50000 \
  --save_interval=500000 \
  --log_interval=50000 \
  --video_episodes=0 \
  $RESUME_ARGS
