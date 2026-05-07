#!/bin/bash -l
#SBATCH --job-name=dgvisc_vis_both
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/dgvisc_vis_both_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/dgvisc_vis_both_%j.err
set -euo pipefail

# Visualize the latest paper-only and paper+geodesic runs and dump both
# contours into one timestamped directory under $HR_DIR/plots/.

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
cd "$DGVISC_DIR"

# If the data-dir env-var patch was lost (fresh re-clone), replay it.
if ! grep -q "DGVISC_DATA_DIR" main.py; then
  echo "=== Re-applying main.py data-dir env-var patch ==="
  python - <<'PYEOF'
import pathlib
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

mkdir -p plots

# Find the latest run sub-directory under exp/goal_representation/<run_group>/
# that has at least one params_*.pkl checkpoint. `exclude` filters run_groups
# whose name contains the given substring (used to keep "geo" runs out of the
# paper-only resolution).
latest_run() {
  local prefix="$1"
  local exclude="${2:-}"
  local grp sub
  while IFS= read -r grp; do
    [ -z "$grp" ] && continue
    grp="${grp%/}"
    if [ -n "$exclude" ] && [[ "$grp" == *"$exclude"* ]]; then
      continue
    fi
    while IFS= read -r sub; do
      [ -z "$sub" ] && continue
      sub="${sub%/}"
      if compgen -G "${sub}/params_*.pkl" > /dev/null; then
        printf '%s\n' "$sub"
        return 0
      fi
    done < <(ls -1dt "${grp}"/*/ 2>/dev/null || true)
  done < <(ls -1dt "exp/goal_representation/${prefix}"*/ 2>/dev/null || true)
  return 0
}

latest_epoch() {
  local run="$1"
  ls -1 "${run}"/params_*.pkl 2>/dev/null \
    | sed -E 's|.*params_([0-9]+)\.pkl|\1|' \
    | sort -n \
    | tail -n 1
}

# Resolve runs.
PAPER_RUN=$(latest_run "dgvisc_ant_med_" "geo")
GEO_RUN=$(latest_run "dgvisc_ant_med_geo_" "")

echo "=== Paper run:  ${PAPER_RUN:-(none)} ==="
echo "=== Geo run:    ${GEO_RUN:-(none)}   ==="

run_vis() {
  local label="$1"; shift
  local run_path="$1"; shift
  if [ -z "$run_path" ]; then
    echo "=== Skipping $label: no run dir resolved ==="
    return 0
  fi
  local epoch
  epoch=$(latest_epoch "$run_path")
  if [ -z "$epoch" ]; then
    echo "=== Skipping $label: no params_*.pkl in $run_path ==="
    return 0
  fi
  echo "=== Visualizing $label (epoch $epoch) :: $run_path ==="
  if python vis.py "$@" --restore_path="$run_path" --restore_epoch="$epoch"; then
    echo "=== $label OK ==="
  else
    local rc=$?
    echo "=== $label FAILED (vis.py exit $rc) — continuing ==="
  fi
  return 0
}

# Common paper config flags shared by both (must match training to load weights).
COMMON=(
  --env_name=antmaze-medium-navigate-v0
  --agent.alpha=10.0
  --agent.rep_type=bilinear
  --agent.rep_expectile=0.9
  --agent.goalrep_dim=256
  --agent.discount=0.99
)

run_vis "Paper (gcivl_dual)" "$PAPER_RUN" \
  --agent=agents/gcivl/state/dual.py \
  "${COMMON[@]}"

run_vis "Paper + HJB (gcivl_dual_geodesic)" "$GEO_RUN" \
  --agent=agents/gcivl/state/dual_geodesic.py \
  --agent.enable_geodesic_hjb=True \
  --agent.geo_kappa=0.01 \
  --agent.w_geo_hjb=0.5 \
  "${COMMON[@]}"

# vis.py writes to $DGVISC_DIR/plots/{Path(restore_path).name}-{save_prefix}.png
# (filenames embed the setting + fk_key + env + exp_name, so paper-only and
# paper+HJB land in distinct files).
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$HR_DIR/plots/dgvisc_vis_$STAMP"
mkdir -p "$OUT_DIR"

# Mirror only the most recent PNGs (those touched in the last hour) so we don't
# scoop up plots from prior vis_both runs.
shopt -s nullglob
PNGS=()
for f in "$DGVISC_DIR/plots"/*.png; do
  if [ "$f" -nt "$DGVISC_DIR/plots/.vis_both_anchor" ] 2>/dev/null || [ ! -e "$DGVISC_DIR/plots/.vis_both_anchor" ]; then
    PNGS+=( "$f" )
  fi
done
shopt -u nullglob

if [ ${#PNGS[@]} -gt 0 ]; then
  cp -v "${PNGS[@]}" "$OUT_DIR/"
  ls -la "$OUT_DIR/"
else
  echo "(no contour plots produced — both vis.py invocations probably errored; check paper/geo run dirs)"
fi

# Drop an anchor file so the next vis_both run mirrors only newer plots.
touch "$DGVISC_DIR/plots/.vis_both_anchor"

echo "=== Done. Plots: $OUT_DIR ==="
