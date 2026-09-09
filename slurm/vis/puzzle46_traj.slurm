#!/bin/bash -l
#SBATCH --job-name=traj_puzzle46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH -o /scratch/gautschi/manors/logs/traj_puzzle46_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/traj_puzzle46_%j.err
set -uo pipefail

source /scratch/gautschi/manors/venvs/ogbench-py311/bin/activate

export BASE=/scratch/gautschi/manors/ogbench
export HR_EXP_DIR=/scratch/gautschi/manors/exp
export HR_DIR=$HOME/horizon-reduction
export PYTHONPATH="$HR_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY

mkdir -p /scratch/gautschi/manors/logs
cd "$HR_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$HR_DIR/plots/traj_$STAMP"
mkdir -p "$OUT_DIR"
echo "=== Output dir: $OUT_DIR ==="

ENV_NAME=puzzle-4x6-play-oraclerep-v0
DATASET_DIR="$BASE/puzzle-4x6-play-1b-v0"
EPOCH=5000000

# Optional positional args override the seeded pass: <tasks> <episode seeds>.
# Seeding the env resets makes a column show the same initial state for every
# method. (These are comma lists, so they cannot travel through sbatch --export.)
SEED_TASKS=${1:-2,3,4}
SEED_LIST=${2:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
NCOLS=5

# The reference method writes its own trace length per panel here; every other
# method then draws the part of its path that runs past that budget in red.
BUDGET="$OUT_DIR/budget.json"
REF_TITLE=phys

FAILED_LABELS=()
SUCCEEDED_LABELS=()

# Args: <label> <group> <title> <agent.py-and-flags...>
run_traj() {
  local label="$1"; shift
  local group="$1"; shift
  local title="$1"; shift
  local path="$HR_EXP_DIR/goal_representation/$group/sd001_*"
  if ! compgen -G "$path" > /dev/null; then
    echo "=== Skipping: $label (no run under $group) ==="
    FAILED_LABELS+=("$label (no run dir)")
    return 0
  fi

  local budget_args=(--budget_json="$BUDGET")
  [ "$title" = "$REF_TITLE" ] && budget_args+=(--write_budget=True)

  # Pass 1 feeds grid_4x5 (one panel per task); pass 2 feeds the fail/success
  # grids, which need a seed to pin the initial state across methods.
  local rc=0
  for pass in plain seeded; do
    local pass_args=(--tasks=1,2,3,4,5)
    [ "$pass" = seeded ] && pass_args=(--tasks="$SEED_TASKS" --episode_seeds="$SEED_LIST")
    echo "=== Rolling out: $label ($group) [$pass] ==="
    python traj_vis.py "$@" \
       "${pass_args[@]}" \
       "${budget_args[@]}" \
       --env_name="$ENV_NAME" \
       --dataset_dir="$DATASET_DIR" \
       --restore_path="$path" \
       --restore_epoch=$EPOCH \
       --cam_elevation=-40 \
       --save_dir="$OUT_DIR/" \
       --title="$title" || rc=$?
  done

  if [ $rc -eq 0 ]; then
    SUCCEEDED_LABELS+=("$label")
  else
    echo "=== FAILED: $label (traj_vis.py exit $rc) — continuing ==="
    FAILED_LABELS+=("$label (exit $rc)")
  fi
  return 0
}

# (c) SHARSA + phys (FK fixed-point). Runs first: it is the reference method, so
# the budget file must exist before anything else is rendered.
run_traj "SHARSA + Phys (FK)" "phys_puzzle46_1b" "phys" \
  --agent=agents/sharsa.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True \
  --agent.w_fk=1.0 \
  --agent.viscous_scale=0.01 \
  --agent.num_walks=10

# (a) Baseline SHARSA
run_traj "Baseline SHARSA" "sharsa_puzzle46_1b" "sharsa" \
  --agent=agents/sharsa.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True

# (b) SHARSA + dual goal
run_traj "SHARSA + Dual" "dual_puzzle46_1b" "dual" \
  --agent=agents/sharsa_dual.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True \
  --agent.rep_type=bilinear \
  --agent.goalrep_dim=256 \
  --agent.rep_expectile=0.9 \
  --agent.rep_w=1.0

# (d) SHARSA + geodesic HJB (isotropic)
run_traj "SHARSA + Geodesic HJB" "geo_puzzle46_1b" "geo" \
  --agent=agents/sharsa_geodesic.py \
  --agent.q_agg=min \
  --agent.subgoal_steps=25 \
  --agent.actor_p_trajgoal=0.5 \
  --agent.actor_p_randomgoal=0.5 \
  --agent.actor_geom_sample=True \
  --agent.use_anisotropic=False \
  --agent.w_geo=1.0 \
  --agent.tightness_threshold=0.1 \
  --agent.metric_rank=8

python traj_montage.py \
  --plot_dir="$OUT_DIR" \
  --methods=sharsa,dual,geo,phys \
  --tasks=1,2,3,4,5 \
  --out="$OUT_DIR/grid_4x5.png"

# Columns for the fail/success grids are picked from this job own outcomes: the
# flow actor is not reproducible across jobs, so a hard-coded (task, seed) list
# from an earlier sweep would not still be a baseline-fails/ours-succeeds case.
# Candidates are spread over tasks before topping up, so the grid is not all
# one task.
read -r COL_TAGS COL_LABELS < <(python - "$OUT_DIR" "$SEED_TASKS" "$SEED_LIST" $NCOLS <<'PY'
import os, sys

out_dir, tasks, seeds, want = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(','), int(sys.argv[4])
cand = [(t, s) for t in tasks for s in seeds
        if os.path.exists(f'{out_dir}/sharsa_task{t}s{s}_success0.png')
        and os.path.exists(f'{out_dir}/phys_task{t}s{s}_success1.png')]

picked = []
for per_task in range(1, want + 1):
    for t in tasks:
        for c in [c for c in cand if c[0] == t][:per_task]:
            if c not in picked and len(picked) < want:
                picked.append(c)
picked.sort(key=lambda c: (int(c[0]), int(c[1])))

if picked:
    print(','.join(f'{t}s{s}' for t, s in picked), ','.join(f'Task {t}' for t, _ in picked))
PY
) || true

if [ -n "${COL_TAGS:-}" ]; then
  NFOUND=$(awk -F, '{print NF}' <<< "$COL_TAGS")
  echo "=== Fail/success columns ($NFOUND): $COL_TAGS ==="
  for methods in sharsa,dual,geo,phys sharsa,phys; do
    n=$(awk -F, '{print NF}' <<< "$methods")
    python traj_montage.py \
      --plot_dir="$OUT_DIR" \
      --methods="$methods" \
      --tasks="$COL_TAGS" \
      --col_labels="$COL_LABELS" \
      --out="$OUT_DIR/grid_failsuccess_${n}x${NFOUND}.png"
  done
else
  echo "=== No baseline-fails/ours-succeeds column found; skipping fail/success grids ==="
fi

echo "=== Trajectory summary ==="
echo "Succeeded: ${SUCCEEDED_LABELS[*]:-(none)}"
echo "Failed:    ${FAILED_LABELS[*]:-(none)}"
ls -la "$OUT_DIR"/*.png 2>/dev/null || echo "(no panels produced)"

if [ ${#FAILED_LABELS[@]} -gt 0 ]; then
  exit 1
fi
