#!/usr/bin/env bash
# One command for the guided Spot session on the Jetson: `python main.py` inside the spot-jax image,
# the only Python with GPU JAX on this machine. After the prompts it follows the logs:
#   Ctrl+C            stops Spot: the running arm saves its buffer and weights, then everything exits
#   closing SSH       leaves the arms running; `spot stop` from any terminal stops them
#   spot --spot_sim   the same flow against the occupancy-grid sim
#   spot stop         stop from another terminal
set -uo pipefail
REPO=/mnt/data/horizon-reduction
NAME=spot-session
STEPS=${SPOT_ONLINE_STEPS:-}
TTY=-i; [ -t 0 ] && TTY=-it
ROOT=/mnt/data/spot_session
case " $* " in *" --spot_sim "*) ROOT=$REPO/exp/spot_session_sim ;; esac

stop_all() {
  trap - INT
  echo
  echo "Stopping Spot..."
  # SIGINT the runner's process group and give the arm 5 s to save, then remove the container.
  # Checkpoints are written every 1000 steps anyway, so a forced stop loses at most the last few.
  sudo docker exec "$NAME" python -c "
import json, os, signal, time
try:
    pid = json.load(open('$ROOT/session.json'))['runner_pid']
    os.killpg(pid, signal.SIGINT)
except Exception as exc:
    print('no runner to stop:', exc)
    raise SystemExit
for _ in range(10):
    try:
        os.kill(pid, 0)
    except OSError:
        break
    time.sleep(0.5)
" || true
  sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "Stopped. Run 'spot' and answer r to resume."
  exit 0
}

if [ "${1:-}" = stop ]; then
  stop_all
fi

# Ctrl+C during the prompts stops too (nothing is running yet, so this just cleans up).
trap stop_all INT
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
exec docker exec $TTY ${STEPS:+-e SPOT_ONLINE_STEPS=$STEPS} $NAME python main.py $*
" || exit $?

# The prompts are done and the arms are running. Follow the logs until Ctrl+C.
trap '' HUP
echo "Following the logs. Ctrl+C stops Spot; closing the terminal leaves it running."
trap stop_all INT
sudo docker exec "$NAME" sh -c "tail -n 5 -F $ROOT/logs/*.log"
stop_all
