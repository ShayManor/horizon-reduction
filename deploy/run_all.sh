#!/usr/bin/env bash
# Run the four arms of the online Spot maze benchmark, one after the other on the one robot.
#
# Usage:
#   export SPOT_IP=192.168.80.3
#   export SPOT_USERNAME=... SPOT_PASSWORD=...      # never flags: flags land in flags.json and wandb
#   deploy/run_all.sh
#
# SSH is only used to start this. Each arm runs under nohup, so the session can go away.
#
# On the Jetson: requirements.txt pins an x86_64 jax[cuda12] wheel. Install NVIDIA's JetPack JAX
# build instead of that line.
#
# The values below are the open decisions in the spec, all in one place. Set them before the first
# run and keep them identical across the four arms: MAX_EPISODE_STEPS in particular decides which
# episodes the steps metric excludes.
set -euo pipefail

: "${SPOT_IP:?set SPOT_IP to the robot address}"
: "${SPOT_USERNAME:?set SPOT_USERNAME}"
: "${SPOT_PASSWORD:?set SPOT_PASSWORD}"

MAP_PATH="${MAP_PATH:-maps/lab}"
ONLINE_STEPS="${ONLINE_STEPS:-200000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
SEED_EPISODES="${SEED_EPISODES:-10}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-1}"
GOAL_TOL="${GOAL_TOL:-0.5}"
VEL_LIMITS="${VEL_LIMITS:-0.6,0.4,0.8}"     # forward, lateral, yaw. Measure these in the lab.
TASK_IDS="${TASK_IDS:-}"                     # empty uses every task in $MAP_PATH/tasks.json
SEED="${SEED:-1}"
LOG_DIR="${LOG_DIR:-exp/spot_logs}"

# Buffer must exceed the whole run so the ring never wraps: past that point trajectory boundaries
# stop being monotone in index and terminal-based slicing breaks.
BUFFER_SIZE="${BUFFER_SIZE:-$((ONLINE_STEPS * 2))}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"

mkdir -p "$LOG_DIR"

common=(
  --env_name=spot-maze-v0
  --online_steps="$ONLINE_STEPS"
  --max_episode_steps="$MAX_EPISODE_STEPS"
  --warmup_steps="$WARMUP_STEPS"
  --seed_episodes="$SEED_EPISODES"
  --updates_per_step="$UPDATES_PER_STEP"
  --buffer_size="$BUFFER_SIZE"
  --control_hz=10
  --goal_tol="$GOAL_TOL"
  --vel_limits="$VEL_LIMITS"
  --graphnav_map_path="$MAP_PATH"
  --robot_hostname="$SPOT_IP"
  --seed="$SEED"
  --save_interval="$SAVE_INTERVAL"
  --log_interval="$LOG_INTERVAL"
  --agent.gc_negative=True
)
if [ -n "$TASK_IDS" ]; then
  common+=(--task_ids="$TASK_IDS")
fi

run_arm() {
  local name="$1"; shift
  local agent_file="$1"; shift
  if [ ! -f "$agent_file" ]; then
    echo "$name: $agent_file is missing" >&2
    exit 1
  fi
  echo "=== $name ==="
  nohup python main.py --agent="$agent_file" --run_group="spot_${name}" "${common[@]}" "$@" \
    > "$LOG_DIR/${name}.log" 2>&1
  echo "=== $name done, log at $LOG_DIR/${name}.log ==="
}

# (a) baseline SHARSA
run_arm sharsa agents/sharsa.py

# (b) SHARSA + dual goal representation
run_arm dual agents/sharsa_dual.py \
  --agent.rep_type=bilinear --agent.goalrep_dim=256 --agent.rep_expectile=0.9 --agent.rep_w=1.0

# (c) SHARSA + phys (stochastic Feynman-Kac)
run_arm phys agents/sharsa.py \
  --agent.w_fk=1.0 --agent.viscous_scale=0.01 --agent.num_walks=10

# (d) SHARSA + geodesic HJB. cost_dims=2 keeps c(s, w) on the xy slice: over the full 7-dim
# observation the cost sums metres, a heading chord and m/s, and the weighting is an accident.
run_arm geo agents/sharsa_geodesic.py \
  --agent.use_anisotropic=False --agent.w_geo=1.0 --agent.use_fk_loss=False \
  --agent.basic_smoothing=False --agent.cost_dims=2
