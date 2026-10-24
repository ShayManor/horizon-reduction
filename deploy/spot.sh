#!/usr/bin/env bash
# One command for the guided Spot session on the Jetson: `python main.py` inside the spot-jax image,
# the only Python with GPU JAX on this machine. The container (spot-session) outlives this command,
# so the four training arms keep running after the prompts finish and SSH goes away.
#   spot              guided session on the robot
#   spot --spot_sim   the same flow against the occupancy-grid sim
#   spot stop         stop the container and everything running in it
set -euo pipefail
REPO=/mnt/data/horizon-reduction
NAME=spot-session
STEPS=${SPOT_ONLINE_STEPS:-}
TTY=-i; [ -t 0 ] && TTY=-it

if [ "${1:-}" = stop ]; then
  # SIGINT the runner's process group so the running arm saves before the container goes away.
  sudo docker exec "$NAME" sh -c 'pid=$(python -c "import json; print(json.load(open(\"/mnt/data/spot_session/session.json\"))[\"runner_pid\"])" 2>/dev/null) && kill -INT -- -$pid 2>/dev/null; sleep 20' || true
  sudo docker rm -f "$NAME"
  exit 0
fi

sudo bash -c "
set -e
source /mnt/data/env.sh
source /mnt/data/spot.env
if ! docker ps --format '{{.Names}}' | grep -qx $NAME; then
  docker rm -f $NAME >/dev/null 2>&1 || true
  docker run -d --init --name $NAME --runtime nvidia --network host \
    -v /mnt/data:/mnt/data -w $REPO \
    -e SPOT_IP -e SPOT_USERNAME -e SPOT_PASSWORD -e WANDB_API_KEY \
    -e WANDB_DIR -e WANDB_CACHE_DIR -e TMPDIR -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    spot-jax:latest sleep infinity >/dev/null
fi
exec docker exec $TTY -e SPOT_ONLINE_STEPS=$STEPS $NAME python main.py $*
"
