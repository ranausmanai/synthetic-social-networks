#!/usr/bin/env bash
# Sync results from the RunPod box to local. Idempotent — uses --update so it
# only transfers files newer than the local copy.
#
# Run anytime: bash pull_results.sh
# Excludes the embedding cache (large, regenerable on the pod).

set -u

POD_USER="${POD_USER:-root}"
POD_HOST="${POD_HOST:?set POD_HOST=<your-pod-ip> in env}"
POD_PORT="${POD_PORT:-22}"
POD_KEY="${POD_KEY:-~/.ssh/id_ed25519}"
POD_ROOT="${POD_ROOT:-/workspace/synthetic-social-networks}"

mkdir -p runs run_logs

echo "[$(date +%H:%M:%S)] pulling runs/ ..."
rsync -rltz --no-perms --no-owner --no-group --update \
  --exclude '.embed_cache/' \
  --exclude '__pycache__/' \
  -e "ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20" \
  "$POD_USER@$POD_HOST:$POD_ROOT/runs/" ./runs/ 2>&1 | tail -3

echo "[$(date +%H:%M:%S)] pulling run_logs/ ..."
rsync -rltz --no-perms --no-owner --no-group --update \
  -e "ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20" \
  "$POD_USER@$POD_HOST:$POD_ROOT/run_logs/" ./run_logs/ 2>&1 | tail -3

echo "[$(date +%H:%M:%S)] sync done."
echo "Local runs:"
du -sh runs/ 2>/dev/null
echo "STATUS:"
cat run_logs/STATUS 2>/dev/null
