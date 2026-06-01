#!/bin/bash -l
#SBATCH --job-name=build_pmlarge_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -A lilly-rob1
#SBATCH -p cpu
#SBATCH --array=0-999%64
#SBATCH -o /scratch/gautschi/manors/logs/build_pmlarge_1b_%A_%a.out
#SBATCH -e /scratch/gautschi/manors/logs/build_pmlarge_1b_%A_%a.err
set -euo pipefail

# Builds pointmaze-large-navigate-1b-v0 via ogbench data_gen_scripts.
# 1000 shards × 1000 trajectories × 1001 steps ≈ 1B transitions.
# Adjust --array=0-N%K to build fewer shards (e.g. 0-99%64 for ~100M) or
# raise concurrency K.

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

# Point this at wherever you cloned seohongpark/ogbench.git (default matches
# the existing checkout under $SCRATCH/ogbench/ogbench).
export OGBENCH_SRC="${OGBENCH_SRC:-/scratch/gautschi/manors/ogbench/ogbench}"
export OUT_DIR=/scratch/gautschi/manors/ogbench/pointmaze-large-navigate-1b-v0

# generate_locomaze.py imports `agents.SACAgent` from impls/; required even for
# non-humanoid envs because the import happens at module top.
export PYTHONPATH="$OGBENCH_SRC/impls:${PYTHONPATH:-}"

mkdir -p /scratch/gautschi/manors/logs "$OUT_DIR"
cd "$OGBENCH_SRC"

SEED=$(printf '%04d' "$SLURM_ARRAY_TASK_ID")
OUT="$OUT_DIR/pointmaze-large-navigate-v0-$SEED.npz"

# Skip if this shard is already on disk (lets you re-run a partial array).
if [ -s "$OUT" ]; then
  echo "=== shard $SEED already exists, skipping ==="
  exit 0
fi

echo "=== Generating shard $SEED -> $OUT ==="
python data_gen_scripts/generate_locomaze.py \
  --seed="$SLURM_ARRAY_TASK_ID" \
  --env_name=pointmaze-large-v0 \
  --save_path="$OUT" \
  --num_episodes=1000 \
  --max_episode_steps=1001 \
  --dataset_type=navigate

echo "=== shard $SEED done ==="
