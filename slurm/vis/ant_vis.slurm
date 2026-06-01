#!/bin/bash -l
#SBATCH --job-name=vis_ant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/vis_ant_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/vis_ant_%j.err
set -uo pipefail

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
export JAX_PLATFORMS=cpu
unset DISPLAY
unset JAX_PLATFORM_NAME

mkdir -p /scratch/gautschi/manors/logs "$HR_EXP_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$HR_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$HR_DIR/plots/$STAMP"
mkdir -p "$OUT_DIR"
echo "=== Output dir: $OUT_DIR ==="

GOAL_IDX=42

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

# Args: <label> <group> <env_name> <dataset_dir> <title> <agent.py-and-flags...>
run_vis() {
  local label="$1"; shift
  local group="$1"; shift
  local env_name="$1"; shift
  local dataset_dir="$1"; shift
  local title="$1"; shift
  local path
  path=$(latest_run "$group")
  if [ -z "$path" ]; then
    echo "=== Skipping: $label (no run found under exp/horizon-reduction/$group) ==="
    FAILED_LABELS+=("$label (no run dir)")
    return 0
  fi
  path="${path%/}"
  echo "=== Visualizing: $label  ($path)  env=$env_name ==="
  if python vis.py "$@" \
       --env_name="$env_name" \
       --dataset_dir="$dataset_dir" \
       --grid_size=100 \
       --goal_idx=$GOAL_IDX \
       --save_dir="$OUT_DIR/" \
       --title="$title" \
       --restore_path="$path"; then
    SUCCEEDED_LABELS+=("$label")
  else
    local rc=$?
    echo "=== FAILED: $label (vis.py exit $rc) — continuing ==="
    FAILED_LABELS+=("$label (vis.py exit $rc)")
  fi
  return 0
}

# The four methods on antmaze-medium 1b.
# Spec format: env_name:dataset_dir:env_tag:group_suffix
ENVS=(
  "antmaze-medium-navigate-v0:$BASE/antmaze-medium-navigate-1b-v0:antmed:ant_1b"
)

for spec in "${ENVS[@]}"; do
  IFS=':' read -r ENV_NAME DATASET_DIR ENV_TAG GROUP_SUFFIX <<< "$spec"

  # (a) Baseline SHARSA
  run_vis "Baseline SHARSA [$ENV_TAG]" \
    "sharsa_${GROUP_SUFFIX}" "$ENV_NAME" "$DATASET_DIR" "sharsa_${ENV_TAG}" \
    --agent=agents/sharsa.py \
    --agent.q_agg=min \
    --agent.subgoal_steps=25

  # (b) SHARSA + dual goal
  run_vis "SHARSA + Dual [$ENV_TAG]" \
    "dual_${GROUP_SUFFIX}" "$ENV_NAME" "$DATASET_DIR" "dual_${ENV_TAG}" \
    --agent=agents/sharsa_dual.py \
    --agent.q_agg=min \
    --agent.subgoal_steps=25 \
    --agent.rep_type=bilinear \
    --agent.goalrep_dim=256 \
    --agent.rep_expectile=0.9 \
    --agent.rep_w=1.0

  # (c) SHARSA + phys (FK fixed-point)
  run_vis "SHARSA + Phys (FK) [$ENV_TAG]" \
    "phys_${GROUP_SUFFIX}" "$ENV_NAME" "$DATASET_DIR" "phys_${ENV_TAG}" \
    --agent=agents/sharsa.py \
    --agent.q_agg=min \
    --agent.subgoal_steps=25 \
    --agent.w_fk=1.0 \
    --agent.viscous_scale=0.01 \
    --agent.num_walks=10

  # (d) SHARSA + geodesic HJB (isotropic)
  run_vis "SHARSA + Geodesic HJB [$ENV_TAG]" \
    "geo_${GROUP_SUFFIX}" "$ENV_NAME" "$DATASET_DIR" "geo_${ENV_TAG}" \
    --agent=agents/sharsa_geodesic.py \
    --agent.q_agg=min \
    --agent.subgoal_steps=25 \
    --agent.use_anisotropic=False \
    --agent.w_geo=1.0 \
    --agent.tightness_threshold=0.1 \
    --agent.metric_rank=8
done

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
    git -c user.name="shaymanor" -c user.email="shay.manor@gmail.com" commit -m "vis: antmaze-medium value contour plots $STAMP" || true
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
