#!/bin/bash -l
#SBATCH --job-name=vis_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/vis_all_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/vis_all_%j.err
set -euo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export BASE=/scratch/gautschi/manors/ogbench
export HR_DIR=$HOME/horizon-reduction
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
unset DISPLAY
unset JAX_PLATFORM_NAME

mkdir -p /scratch/gautschi/manors/logs
cd "$HR_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$HR_DIR/plots/$STAMP"
mkdir -p "$OUT_DIR"
echo "=== Output dir: $OUT_DIR ==="

ENV_NAME="antmaze-medium-navigate-v0"
DATASET_DIR="$BASE/antmaze-medium-navigate-1b-v0"

GOAL_IDX=42
COMMON="--env_name=$ENV_NAME --dataset_dir=$DATASET_DIR --grid_size=100 --goal_idx=$GOAL_IDX --save_dir=$OUT_DIR/"

# Resolve the most recent training run directory under a run_group (by mtime),
# considering only runs that have at least one params_*.pkl checkpoint saved.
latest_run() {
  local group="$1"
  local d
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    if compgen -G "${d}params_*.pkl" > /dev/null; then
      printf '%s\n' "$d"
      return 0
    fi
  done < <(ls -1dt "exp/horizon-reduction/$group"/sd*/ 2>/dev/null || true)
  return 0
}

FAILED_LABELS=()
SUCCEEDED_LABELS=()

run_vis() {
  local label="$1"; shift
  local group="$1"; shift
  local path
  path=$(latest_run "$group")
  if [ -z "$path" ]; then
    echo "=== Skipping: $label (no run found under exp/horizon-reduction/$group) ==="
    FAILED_LABELS+=("$label (no run dir)")
    return 0
  fi
  path="${path%/}"
  echo "=== Visualizing: $label  ($path) ==="
  if python vis.py "$@" --restore_path="$path" $COMMON; then
    SUCCEEDED_LABELS+=("$label")
  else
    local rc=$?
    echo "=== FAILED: $label (vis.py exit $rc) — continuing ==="
    FAILED_LABELS+=("$label (vis.py exit $rc)")
  fi
  return 0
}

# Baseline SHARSA
run_vis "Baseline SHARSA" \
  "baseline_sharsa_ant_1b" \
  --agent=agents/sharsa.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --title=baseline_sharsa

# Basic Smoothing (skipped if not trained yet)
run_vis "Basic Smoothing" \
  "basic_smoothing_ant_1b" \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.basic_smoothing=True \
  --agent.w_geo=0.5 \
  --title=basic_smoothing

# Geodesic (isotropic)
run_vis "Geodesic Isotropic" \
  "geo_aniso_ant_1b_dev" \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.use_anisotropic=False \
  --agent.w_geo=1.0 \
  --title=sharsa_geodesic

# FK (stochastic Feynman-Kac viscous regularizer)
run_vis "FK" \
  "fk_ant_1b" \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.use_fk_loss=True \
  --agent.viscous_scale=0.01 \
  --agent.num_walks=10 \
  --agent.w_geo=1.0 \
  --title=fk

echo "=== Visualization summary ==="
echo "Succeeded: ${SUCCEEDED_LABELS[*]:-(none)}"
echo "Failed:    ${FAILED_LABELS[*]:-(none)}"
ls -la "$OUT_DIR"/*.png 2>/dev/null || echo "(no plots produced)"

# Commit and push whatever plots did get produced. Always run, even if some methods failed.
echo "=== Committing plots to git ==="
cd "$HR_DIR"
shopt -s nullglob
PLOTS=( "plots/$STAMP"/*.png )
shopt -u nullglob
if [ ${#PLOTS[@]} -eq 0 ]; then
  echo "No plots produced; nothing to commit."
else
  git add "${PLOTS[@]}" || true
  if git diff --cached --quiet; then
    echo "No plot changes to commit."
  else
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    # Compute nodes have no askpass tty; bypass HTTPS entirely by talking to the
    # SSH URL directly. Avoids any reliance on origin's URL or insteadOf rewrites.
    SSH_URL="git@github.com:ShayManor/horizon-reduction.git"
    export GIT_TERMINAL_PROMPT=0
    export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    git -c user.name="shaymanor" -c user.email="shay.manor@gmail.com" commit -m "vis: value contour plots $STAMP" || true
    git pull --rebase --autostash "$SSH_URL" "$BRANCH" || echo "git pull/rebase failed (continuing to push attempt)"
    if git push "$SSH_URL" "HEAD:$BRANCH"; then
      echo "git push OK"
    else
      echo "git push failed"
    fi
  fi
fi

# Exit non-zero if any method failed, but only AFTER the push has run.
if [ ${#FAILED_LABELS[@]} -gt 0 ]; then
  echo "=== Some methods failed: ${FAILED_LABELS[*]} ==="
  exit 1
fi