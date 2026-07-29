#!/usr/bin/env bash
set -euo pipefail

cd /workspace/socialresearch2
mkdir -p /workspace/logs /workspace/socialresearch2/runs_confirmatory

source /workspace/venv/bin/activate
export OLLAMA_MODELS=/workspace/ollama-models

stamp=$(date +%Y%m%d-%H%M%S)
log="/workspace/logs/confirmatory_${stamp}.log"
deadline_epoch=$(( $(date +%s) + 18 * 3600 ))

nohup bash -lc "
cd /workspace/socialresearch2
source /workspace/venv/bin/activate
export OLLAMA_MODELS=/workspace/ollama-models
python -m src.experiments.confirmatory \
  --config configs/confirmatory_v1.yaml \
  --output runs_confirmatory/v1_core \
  --deadline-epoch $deadline_epoch
python -m src.experiments.confirmatory \
  --config configs/confirmatory_large_v1.yaml \
  --output runs_confirmatory/v1_size_extension \
  --deadline-epoch $deadline_epoch
python -m src.analyze_confirmatory runs_confirmatory/v1_core || true
python -m src.analyze_confirmatory runs_confirmatory/v1_size_extension || true
" \
  >"$log" 2>&1 </dev/null &

echo "$!" >/workspace/logs/confirmatory.pid
echo "$log" >/workspace/logs/confirmatory.current
echo "PID $!"
echo "LOG $log"
