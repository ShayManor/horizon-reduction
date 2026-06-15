#!/bin/bash
# Helper for chained slurm jobs.
#
# Usage in a training slurm script (after HR_EXP_DIR and HR_DIR are exported):
#   source "$HR_DIR/slurm/_resume.sh"
#   apply_resume_args "<run_group>"
#   python main.py ... \
#     $RESUME_ARGS
#
# Behavior:
#   - If env var RESUME != "true", does nothing (fresh run).
#   - If RESUME=true: finds the most recent run directory under
#     $HR_EXP_DIR/goal_representation/<run_group>/sd*/ that has a
#     params_*.pkl checkpoint, picks the largest step number, and sets
#     RESUME_ARGS="--restore_path=<that_dir> --restore_epoch=<that_step>".
#   - If RESUME=true but no checkpoint found, prints a warning and runs fresh.

apply_resume_args() {
  RESUME_ARGS=""
  local run_group="$1"
  if [ "${RESUME:-}" != "true" ]; then
    return 0
  fi
  if [ -z "${HR_EXP_DIR:-}" ]; then
    echo "=== RESUME=true but HR_EXP_DIR is unset; running fresh ==="
    return 0
  fi
  local search_dir="$HR_EXP_DIR/goal_representation/$run_group"
  local latest_dir=""
  local d
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    if compgen -G "${d}params_*.pkl" > /dev/null; then
      latest_dir="${d%/}"
      break
    fi
  done < <(ls -1dt "$search_dir"/sd$(printf '%03d' "${SEED:-1}")*/ 2>/dev/null || true)

  if [ -z "$latest_dir" ]; then
    echo "=== RESUME=true but no checkpoints found under $search_dir; running fresh ==="
    return 0
  fi

  local latest_step
  latest_step=$(ls "$latest_dir"/params_*.pkl 2>/dev/null \
    | sed -n 's:.*/params_\([0-9][0-9]*\)\.pkl$:\1:p' \
    | sort -n | tail -1)

  if [ -z "$latest_step" ]; then
    echo "=== RESUME=true but failed to parse step number under $latest_dir; running fresh ==="
    return 0
  fi

  echo "=== RESUMING $run_group from $latest_dir @ step $latest_step ==="
  RESUME_ARGS="--restore_path=$latest_dir --restore_epoch=$latest_step"
}
