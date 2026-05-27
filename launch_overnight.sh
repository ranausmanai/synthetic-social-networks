#!/usr/bin/env bash
# Launches the killer-research overnight sequence on the pod.
# Order: EXP-2 (headline) → EXP-6 (interventions) → EXP-1 (baseline).
# Each one nohup'd so it survives ssh disconnect.

set -u
cd /workspace/socialresearch2

LOGDIR=/workspace/socialresearch2/run_logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d-%H%M%S)

# Sequential chain — Ollama serializes per-model anyway, so concurrent runs
# would just queue and slow each other.
nohup bash -c "
  echo '=== [A1 anchor start] \$(date) ===' &&
  python3 -m src.run_experiment --exp anchor --config configs/anchor.yaml --tag a1 &&
  echo '=== [A1 done] \$(date) ===' &&
  echo '=== [A2 human-replay start] \$(date) ===' &&
  python3 -m src.run_experiment --exp human_replay --config configs/human_replay.yaml --tag a2 &&
  echo '=== [A2 done] \$(date) ===' &&
  echo '=== [EXP-2 astroturf start] \$(date) ===' &&
  python3 -m src.run_experiment --exp astroturf --config configs/exp2_astroturf.yaml --tag exp2 &&
  echo '=== [EXP-2 done] \$(date) ===' &&
  echo '=== [EXP-6 misinfo start] \$(date) ===' &&
  python3 -m src.run_experiment --exp misinfo --config configs/exp6_misinfo.yaml --tag exp6 &&
  echo '=== [EXP-6 done] \$(date) ===' &&
  echo '=== [EXP-1 baseline start] \$(date) ===' &&
  python3 -m src.run_experiment --sweep --config configs/exp1_baseline.yaml --tag exp1 &&
  echo '=== [EXP-1 done] \$(date) ===' &&
  echo '=== ALL DONE \$(date) ==='
" > "$LOGDIR/overnight_$STAMP.log" 2>&1 < /dev/null &
disown

echo "Launched as PID $!"
echo "Log: $LOGDIR/overnight_$STAMP.log"
echo
echo "Monitor with:"
echo "  ssh \$POD_USER@\$POD_HOST -p \$POD_PORT -i \$POD_KEY \\"
echo "    'tail -f /workspace/socialresearch2/run_logs/overnight_$STAMP.log'"
