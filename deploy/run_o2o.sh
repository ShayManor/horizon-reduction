#!/usr/bin/env bash
# Warm-start one arm from its offline checkpoint and keep training it online.
#
# Usage:
#   RESTORE_PATH=exp/goal_representation/spot_phys/sd001_s_16560237.0.20260922_002619 \
#   RESTORE_EPOCH=8000 DATASET_DIR=exp/spot_offline deploy/run_o2o.sh phys
#
# ROBOT defaults to `sim`, which runs against $MAP_PATH/occupancy.npz instead of the robot. Set
# ROBOT=$SPOT_IP (plus SPOT_USERNAME/SPOT_PASSWORD, never flags) to run on Spot.
#
# The offline run saved weights and no buffer, so the buffer is seeded from $DATASET_DIR: the
# dataset the restored weights were trained on. ONLINE_STEPS counts steps taken online, on top of
# those seeded transitions.
#
# The agent flags below have to match the offline run or the restore hits a shape mismatch. They
# are the ones that differ from the agent defaults in that run's flags.json.
set -euo pipefail

ARM="${1:?usage: deploy/run_o2o.sh sharsa|dual|phys|geo}"

: "${RESTORE_PATH:?set RESTORE_PATH to the offline run directory}"
: "${DATASET_DIR:?set DATASET_DIR to the offline dataset the checkpoint was trained on}"
RESTORE_EPOCH="${RESTORE_EPOCH:-8000}"

ROBOT="${ROBOT:-sim}"
MAP_PATH="${MAP_PATH:-spot_data}"
ONLINE_STEPS="${ONLINE_STEPS:-10000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-1}"
GOAL_TOL="${GOAL_TOL:-0.5}"
VEL_LIMITS="${VEL_LIMITS:-0.6,0.4,0.8}"
SEED="${SEED:-1}"
LOG_DIR="${LOG_DIR:-exp/spot_logs}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"

# The seeded offline transitions sit in the buffer alongside the online ones, so capacity has to
# cover both. Past capacity the ring wraps, trajectory boundaries stop being monotone in index,
# and terminal-based slicing breaks. This counts the whole dataset; the buffer is seeded with the
# training split that the offline run used, which is 5% smaller, so the sizing has slack.
OFFLINE_SIZE=$(python -c "
import glob, numpy as np
print(sum(len(np.load(f)['terminals']) for f in sorted(glob.glob('$DATASET_DIR/*.npz')) if '-val.npz' not in f))
")
BUFFER_SIZE="${BUFFER_SIZE:-$((OFFLINE_SIZE + ONLINE_STEPS + 10000))}"

# Warmup is already satisfied by the seeded buffer, so updates start on the first online step.
WARMUP_STEPS="${WARMUP_STEPS:-1}"

mkdir -p "$LOG_DIR"

common=(
  --env_name=spot-maze-v0
  --restore_path="$RESTORE_PATH"
  --restore_epoch="$RESTORE_EPOCH"
  --dataset_dir="$DATASET_DIR"
  --online_steps="$ONLINE_STEPS"
  --max_episode_steps="$MAX_EPISODE_STEPS"
  --warmup_steps="$WARMUP_STEPS"
  --seed_episodes=0
  --updates_per_step="$UPDATES_PER_STEP"
  --buffer_size="$BUFFER_SIZE"
  --control_hz=10
  --goal_tol="$GOAL_TOL"
  --vel_limits="$VEL_LIMITS"
  --graphnav_map_path="$MAP_PATH"
  --robot_hostname="$ROBOT"
  --seed="$SEED"
  --save_interval="$SAVE_INTERVAL"
  --log_interval="$LOG_INTERVAL"
  --agent.gc_negative=True
  --agent.value_loss_type=squared
  --agent.subgoal_steps=10
)

case "$ARM" in
  sharsa) agent=agents/sharsa.py; extra=() ;;
  dual)   agent=agents/sharsa_dual.py
          extra=(--agent.rep_type=bilinear --agent.goalrep_dim=256
                 --agent.rep_expectile=0.9 --agent.rep_w=1.0) ;;
  phys)   agent=agents/sharsa.py
          extra=(--agent.w_fk=1.0 --agent.viscous_scale=0.01 --agent.num_walks=10
                 --agent.fk_kappa=15.0) ;;
  geo)    agent=agents/sharsa_geodesic.py
          extra=(--agent.use_anisotropic=False --agent.w_geo=1.0 --agent.kappa=16.7
                 --agent.use_fk_loss=False --agent.basic_smoothing=False --agent.cost_dims=2) ;;
  *) echo "unknown arm $ARM" >&2; exit 1 ;;
esac

echo "=== $ARM: warm start from $RESTORE_PATH @ $RESTORE_EPOCH against $ROBOT ==="
echo "    buffer $BUFFER_SIZE covers $OFFLINE_SIZE dataset + $ONLINE_STEPS online + headroom"
nohup python main.py --agent="$agent" --run_group="spot_o2o_${ARM}" "${common[@]}" "${extra[@]}" \
  > "$LOG_DIR/o2o_${ARM}.log" 2>&1
echo "=== $ARM done, log at $LOG_DIR/o2o_${ARM}.log ==="
