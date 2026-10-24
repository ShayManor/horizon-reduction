#!/usr/bin/env bash
# All four methods over every task, on the robot or in the simulator.
#
#   ROBOT=$SPOT_IP deploy/run_benchmark.sh          # on Spot
#   ROBOT=sim      deploy/run_benchmark.sh          # occupancy grid, no robot
#
# One process per method, so the ~55 s JIT compile is paid four times rather than once per task.
# Each method writes its own CSV to $OUT, and the agent flags below are the ones that differ from
# each agent's defaults in that run's flags.json. They must match or the restore hits a shape
# mismatch.
set -euo pipefail

ROBOT="${ROBOT:-sim}"
EPISODES="${EPISODES:-5}"
TASK_IDS="${TASK_IDS:-1,2,3,4,5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
EPOCH="${EPOCH:-8000}"
OUT="${OUT:-/workspace/benchmark}"
MAP_PATH="${MAP_PATH:-spot_data}"
BASE="${BASE:-exp/goal_representation}"
# Which methods to run, so a run interrupted partway can resume without repeating what is done.
METHODS="${METHODS:-sharsa dual phys geo}"
# RESUME=1 skips task/episode pairs already recorded in each method's CSV, so an interrupted
# benchmark picks up where it stopped instead of repeating completed episodes.
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

mkdir -p "$OUT"

COMMON="--gc_negative=True --value_loss_type=squared --subgoal_steps=10"

# Each method runs in its own process, and a process that exits leaves its e-stop keepalive
# policy registered. Check-ins stop, the 9 s timer fires, and the next method cannot power on.
# Clearing the lapsed policies before each method is what lets the four run unattended.
clear_stale_estop() {
  if [ "$ROBOT" != "sim" ]; then
    python deploy/clear_estop.py --robot_hostname="$ROBOT" || true
  fi
}

run_method() {
  local name="$1" agent="$2" ckpt="$3"; shift 3
  echo
  echo "################ $name ################"
  clear_stale_estop
  local args=()
  for f in $COMMON; do args+=("--agent.${f#--}"); done
  for f in "$@"; do args+=("--agent.$f"); done
  python deploy/run_task.py \
    --agent="$agent" \
    --restore_path="$BASE/$ckpt" \
    --restore_epoch="$EPOCH" \
    --robot_hostname="$ROBOT" \
    --graphnav_map_path="$MAP_PATH" \
    --task_ids="$TASK_IDS" \
    --episodes="$EPISODES" \
    --max_episode_steps="$MAX_EPISODE_STEPS" \
    --method="$name" \
    --log_path="$OUT/$name.csv" \
    $RESUME_FLAG \
    "${args[@]}"
}

for method in $METHODS; do
  case "$method" in
    sharsa)
      run_method sharsa agents/sharsa.py \
        spot_sharsa/sd001_s_16560235.0.20260921_230757 ;;
    dual)
      run_method dual agents/sharsa_dual.py \
        spot_dual/sd001_s_16560236.0.20260921_230909 \
        rep_type=bilinear goalrep_dim=256 rep_expectile=0.9 rep_w=1.0 ;;
    phys)
      run_method phys agents/sharsa.py \
        spot_phys/sd001_s_16560237.0.20260922_002619 \
        w_fk=1.0 viscous_scale=0.01 num_walks=10 fk_kappa=15.0 ;;
    geo)
      run_method geo agents/sharsa_geodesic.py \
        spot_geo/sd001_s_16560238.0.20260922_002817 \
        use_anisotropic=False w_geo=1.0 kappa=16.7 use_fk_loss=False basic_smoothing=False cost_dims=2 ;;
    *)
      echo "unknown method $method" >&2; exit 1 ;;
  esac
done

echo
echo "CSVs in $OUT"
