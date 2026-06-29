#!/bin/bash -l
#SBATCH --job-name=ogbench_1b_puzzle46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/ogbench_1b_puzzle46_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/ogbench_1b_puzzle46_%j.err
set -euo pipefail
source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export OGBENCH_SRC=/scratch/gautschi/manors/ogbench-src
export OUT_DIR=/scratch/gautschi/manors/ogbench/puzzle-4x6-play-1b-v0
# puzzle = manipspace, scripted oracle: no expert policy, no GPU, no impls needed.
export PYTHONPATH="$OGBENCH_SRC"
export JAX_PLATFORMS=cpu

mkdir -p /scratch/gautschi/manors/logs "$OUT_DIR"
cd "$OGBENCH_SRC/data_gen_scripts"

# README loop: puzzle-4x6-play-1b = 1000 shards * 1000 length-1001 play trajectories.
gen_one() {
  local i="$1"
  local out="$OUT_DIR/puzzle-4x6-play-v0-$i.npz"
  if [ -s "$out" ]; then return 0; fi
  python generate_manipspace.py \
    --seed="$i" \
    --env_name=puzzle-4x6-v0 \
    --save_path="$out" \
    --num_episodes=1000 \
    --max_episode_steps=1001 \
    --dataset_type=play
}
export -f gen_one
export OUT_DIR

seq -w 0 999 | xargs -n 1 -P "${SLURM_CPUS_PER_TASK:-64}" -I{} bash -c 'gen_one "$@"' _ {}

ls "$OUT_DIR"/*.npz | grep -v -- -val | wc -l
du -sh "$OUT_DIR"
