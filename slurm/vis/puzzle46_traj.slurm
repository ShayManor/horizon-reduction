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
  echo "=== Rolling out: $label ($group) ==="
  if python traj_vis.py "$@" \
       --env_name="$ENV_NAME" \
       --dataset_dir="$DATASET_DIR" \
       --restore_path="$path" \
       --restore_epoch=$EPOCH \
       --cam_elevation=-40 \
       --save_dir="$OUT_DIR/" \
       --title="$title"; then
    SUCCEEDED_LABELS+=("$label")
  else
    local rc=$?
    echo "=== FAILED: $label (traj_vis.py exit $rc) — continuing ==="
    FAILED_LABELS+=("$label (exit $rc)")
  fi
  return 0
}

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

# (c) SHARSA + phys (FK fixed-point)
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

echo "=== Trajectory summary ==="
echo "Succeeded: ${SUCCEEDED_LABELS[*]:-(none)}"
echo "Failed:    ${FAILED_LABELS[*]:-(none)}"
ls -la "$OUT_DIR"/*.png 2>/dev/null || echo "(no panels produced)"

if [ ${#FAILED_LABELS[@]} -gt 0 ]; then
  exit 1
fi
