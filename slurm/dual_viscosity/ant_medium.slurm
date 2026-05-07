#!/bin/bash -l
#SBATCH --job-name=dgvisc_ant_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/dgvisc_ant_med_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/dgvisc_ant_med_%j.err
set -euo pipefail

# Paper's stack: agents/gcivl/state/dual.py — GCIVL + Dual goal rep + FK regularizer.
# Goal of this job: reproduce the paper's antmaze-medium contour on Gautschi.
# We do NOT graft anything here onto our SHARSA agent; that experiment lives in
# slurm/fk/ and is known to not reproduce the paper for reasons documented in

# Try the existing ogbench-py311 venv. If imports fail at runtime (e.g. distrax,
# matplotlib==3.7.5 mismatches per dual-goal-viscosity/README.md), create a
# dedicated dgvisc venv per the paper's README and point this script at it.
source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export HR_DIR=$HOME/horizon-reduction
export DGVISC_DIR="$HR_DIR/dual-goal-viscosity"

# Read by the patched dual-goal-viscosity/main.py (line ~76, env-var fallthrough).
# ogbench will download antmaze-medium-navigate-v0 here on first use.
export DGVISC_DATA_DIR=/scratch/gautschi/manors/ogbench_dgvisc/

export PYTHONPATH="$DGVISC_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

mkdir -p /scratch/gautschi/manors/logs
mkdir -p "$DGVISC_DATA_DIR"

cd "$DGVISC_DIR"

RUN_GROUP="dgvisc_ant_med_$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID"

python main.py \
  --run_group="$RUN_GROUP" \
  --seed=0 \
  --env_name=antmaze-medium-navigate-v0 \
  --agent=agents/gcivl/state/dual.py \
  --agent.alpha=10.0 \
  --agent.rep_type=bilinear \
  --agent.rep_expectile=0.9 \
  --agent.goalrep_dim=256 \
  --agent.discount=0.99 \
  --train_steps=1000000 \
  --eval_interval=250000 \
  --save_interval=500000 \
  --log_interval=5000 \
  --video_episodes=0 \
  --eval_episodes=20

# Path layout from main.py:67 — exp/<project>/<run_group>/<setting>_<fk_key><env_name>_<exp_name>
RUN_DIR=$(ls -1dt "$DGVISC_DIR/exp/goal_representation/$RUN_GROUP"/*/ 2>/dev/null | head -n 1 | sed 's:/$::')
if [ -z "$RUN_DIR" ]; then
  echo "=== FAILED: no run dir under exp/goal_representation/$RUN_GROUP/ ==="
  exit 1
fi
echo "=== Trained run dir: $RUN_DIR ==="

# Generate the contour with the paper's vis.py
# vis.py writes to plots/{Path(restore_path).name}-{save_prefix}.png in cwd.
mkdir -p "$DGVISC_DIR/plots"
python vis.py \
  --env_name=antmaze-medium-navigate-v0 \
  --agent=agents/gcivl/state/dual.py \
  --agent.alpha=10.0 \
  --agent.rep_type=bilinear \
  --agent.rep_expectile=0.9 \
  --agent.goalrep_dim=256 \
  --agent.discount=0.99 \
  --restore_path="$RUN_DIR" \
  --restore_epoch=1000000 || echo "vis.py exited non-zero (training output is still preserved at $RUN_DIR)"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$HR_DIR/plots/dgvisc_$STAMP"
mkdir -p "$OUT_DIR"
shopt -s nullglob
PNGS=( "$DGVISC_DIR/plots"/*.png )
shopt -u nullglob
if [ ${#PNGS[@]} -gt 0 ]; then
  cp -v "${PNGS[@]}" "$OUT_DIR/"
  ls -la "$OUT_DIR/"
else
  echo "(no contour plots produced — vis.py probably errored)"
fi

echo "=== Done. Run dir: $RUN_DIR ==="
echo "=== Plots mirrored to: $OUT_DIR ==="
