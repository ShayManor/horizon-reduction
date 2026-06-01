#!/bin/bash -l
#SBATCH --job-name=build_pmlarge_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/build_pmlarge_1b_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/build_pmlarge_1b_%j.err
set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export OGBENCH_SRC=/scratch/gautschi/manors/ogbench/ogbench
export OUT_DIR=/scratch/gautschi/manors/ogbench/pointmaze-large-navigate-1b-v0
export PYTHONPATH="$OGBENCH_SRC/impls"

mkdir -p /scratch/gautschi/manors/logs "$OUT_DIR"
cd "$OGBENCH_SRC"

gen_one() {
  local i="$1"
  local seed_str
  seed_str=$(printf '%04d' "$i")
  local out="$OUT_DIR/pointmaze-large-navigate-v0-$seed_str.npz"
  if [ -s "$out" ]; then return 0; fi
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

seq 0 999 | xargs -n 1 -P 64 -I{} bash -c 'gen_one "$@"' _ {}

ls "$OUT_DIR" | wc -l
du -sh "$OUT_DIR"
