#!/bin/bash -l
#SBATCH --job-name=build_pmlarge_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p cpu
#SBATCH -o /scratch/gautschi/manors/logs/build_pmlarge_1b_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/build_pmlarge_1b_%j.err
set -euo pipefail

# Builds pointmaze-large-navigate-1b-v0 via ogbench data_gen_scripts in a
# single SLURM job. 1000 shards × 1000 trajectories × 1001 steps ≈ 1B
# transitions, parallelized 64-way via xargs.
# To build fewer shards (e.g. ~100M for a sanity comparison), reduce SHARDS.

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

SHARDS=1000
PARALLEL=64

gen_one() {
  local i="$1"
  local seed_str
  seed_str=$(printf '%04d' "$i")
  local out="$OUT_DIR/pointmaze-large-navigate-v0-$seed_str.npz"
  if [ -s "$out" ]; then
    echo "[$seed_str] already exists, skipping"
    return 0
  fi
  echo "[$seed_str] generating -> $out"
  python data_gen_scripts/generate_locomaze.py \
    --seed="$i" \
    --env_name=pointmaze-large-v0 \
    --save_path="$out" \
    --num_episodes=1000 \
    --max_episode_steps=1001 \
    --dataset_type=navigate
}
export -f gen_one
export OUT_DIR

seq 0 $((SHARDS - 1)) \
  | xargs -n 1 -P "$PARALLEL" -I{} bash -c 'gen_one "$@"' _ {}

echo "=== Build complete ==="
ls "$OUT_DIR" | wc -l
du -sh "$OUT_DIR"
