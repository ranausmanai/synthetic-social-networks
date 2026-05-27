#!/usr/bin/env bash
# Calibration chain — runs after the headline experiments finish.
# Order: A1 bandwagon anchor → A2 human-replay (synthetic dataset by default).
#
# These give the paper its methodological credibility — they reproduce a known
# human result (A1) and predict real human view-shifts on ChangeMyView (A2).

set -u
cd /workspace/socialresearch2

LOGDIR=/workspace/socialresearch2/run_logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d-%H%M%S)

nohup bash -c "
  echo '=== [A1 anchor start] \$(date) ===' &&
  python3 -m src.run_experiment --exp anchor --config configs/anchor.yaml --tag a1 &&
  echo '=== [A1 done] \$(date) ===' &&
  echo '=== [A2 human-replay start] \$(date) ===' &&
  python3 -m src.run_experiment --exp human_replay --config configs/human_replay.yaml --tag a2 &&
  echo '=== [A2 done] \$(date) ===' &&
  echo '=== [EXP-3 influence start] \$(date) ===' &&
  python3 -m src.run_experiment --exp influence --config configs/exp3_influence.yaml --tag exp3 &&
  echo '=== [EXP-3 done] \$(date) ===' &&
  echo '=== CALIBRATION DONE \$(date) ==='
" > "$LOGDIR/calibration_$STAMP.log" 2>&1 < /dev/null &
disown

echo "Launched calibration as PID \$!"
echo "Log: $LOGDIR/calibration_$STAMP.log"
