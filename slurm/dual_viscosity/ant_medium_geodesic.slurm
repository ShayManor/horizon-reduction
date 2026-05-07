#!/bin/bash -l
#SBATCH --job-name=dgvisc_ant_med_geo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/dgvisc_ant_med_geo_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/dgvisc_ant_med_geo_%j.err
set -euo pipefail

# gcivl_dual stack + our geodesic-HJB regularizer added on top.
# Agent: agents/gcivl/state/dual_geodesic.py
#   = paper's rep_loss + value_loss(with FK) + actor_loss + geodesic_hjb_loss
source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export HR_DIR=$HOME/horizon-reduction
export DGVISC_DIR="$HR_DIR/dual-goal-viscosity"

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

# If the data-dir env-var patch was lost (e.g. fresh re-clone of the upstream),
# replay it idempotently before launching python.
if ! grep -q "DGVISC_DATA_DIR" main.py; then
  echo "=== Re-applying main.py data-dir env-var patch ==="
  python - <<'PYEOF'
import re, pathlib
p = pathlib.Path('main.py')
src = p.read_text()
old = "env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'], dataset_dir='/scratch/gautschi/hviswan/ogbench/data/')"
new = (
    "_data_dir = os.environ.get('DGVISC_DATA_DIR', '/scratch/gautschi/hviswan/ogbench/data/')\n"
    "    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'], dataset_dir=_data_dir)"
)
assert old in src, "expected upstream main.py line not found; manual patch needed"
p.write_text(src.replace(old, new))
print("patched")
PYEOF
fi

RUN_GROUP="dgvisc_ant_med_geo_$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID"

python main.py \
  --run_group="$RUN_GROUP" \
  --seed=0 \
  --env_name=antmaze-medium-navigate-v0 \
  --agent=agents/gcivl/state/dual_geodesic.py \
  --agent.alpha=10.0 \
  --agent.rep_type=bilinear \
  --agent.rep_expectile=0.9 \
  --agent.goalrep_dim=256 \
  --agent.discount=0.99 \
  --agent.enable_geodesic_hjb=True \
  --agent.geo_kappa=0.01 \
  --agent.w_geo_hjb=0.5 \
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

# Generate the contour with the paper's vis.py.
# vis.py writes to plots/{Path(restore_path).name}-{save_prefix}.png in cwd.
mkdir -p "$DGVISC_DIR/plots"
python vis.py \
  --env_name=antmaze-medium-navigate-v0 \
  --agent=agents/gcivl/state/dual_geodesic.py \
  --agent.alpha=10.0 \
  --agent.rep_type=bilinear \
  --agent.rep_expectile=0.9 \
  --agent.goalrep_dim=256 \
  --agent.discount=0.99 \
  --agent.enable_geodesic_hjb=True \
  --agent.geo_kappa=0.01 \
  --agent.w_geo_hjb=0.5 \
  --restore_path="$RUN_DIR" \
  --restore_epoch=1000000 || echo "vis.py exited non-zero (training output is still preserved at $RUN_DIR)"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$HR_DIR/plots/dgvisc_geo_$STAMP"
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
