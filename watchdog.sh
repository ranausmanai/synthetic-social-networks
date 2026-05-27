#!/usr/bin/env bash
# Pod-side SELF-HEALING watchdog v2 — bulletproof pgrep + restart safety buffer.
#
# Behavior:
#   - heartbeats every 90s
#   - only declares chain dead after 3 CONSECUTIVE "no process" checks (~4.5 min)
#   - max 2 auto-restarts of the main chain
#   - auto-launches calibration once main chain ALL DONE
#   - exits when calibration ALL DONE or restart-cap hit
#
# Files (in /workspace/socialresearch2/run_logs/):
#   heartbeat.txt          append-only timeline
#   STATUS                 single-line current state (tail this from phone)
#   CRASH_DETECTED         appears only after restart cap hit
#   FINAL_DONE             appears when both main + calibration finished
#   CALIBRATION_LAUNCHED   marker so calibration only kicks off once

set -u

ROOT=/workspace/socialresearch2
LOGDIR="$ROOT/run_logs"
HEARTBEAT="$LOGDIR/heartbeat.txt"
STATUS="$LOGDIR/STATUS"
CRASH="$LOGDIR/CRASH_DETECTED"
FINAL_DONE="$LOGDIR/FINAL_DONE"
CALIBRATION_LAUNCHED="$LOGDIR/CALIBRATION_LAUNCHED"
RESTART_COUNT_FILE="$LOGDIR/restart_count"
DEAD_CHECKS_FILE="$LOGDIR/dead_checks"

mkdir -p "$LOGDIR"

MAX_RESTARTS=2
DEAD_CHECKS_THRESHOLD=3   # number of consecutive dead checks before restart
SLEEP_SEC=90

[ -f "$RESTART_COUNT_FILE" ] || echo 0 > "$RESTART_COUNT_FILE"
[ -f "$DEAD_CHECKS_FILE" ] || echo 0 > "$DEAD_CHECKS_FILE"

write_status() {
    echo "$1" > "$STATUS"
}

heartbeat() {
    local now status gpu calls csv_rows
    now=$(date -Is)
    status="$1"
    gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %')
    calls=$(grep -c "POST.*api/generate" /workspace/ollama.log 2>/dev/null || echo 0)
    csv_rows=$(find /workspace/socialresearch2/runs -name "*_results.csv" -exec wc -l {} \; 2>/dev/null | awk '{s+=$1} END {print s+0}')
    echo "[$now] $status | gpu=${gpu}% | ollama_calls=$calls | csv_rows=$csv_rows" >> "$HEARTBEAT"
    write_status "[$now] $status (gpu=${gpu}%, calls=$calls, csv_rows=$csv_rows)"
}

# Three SEPARATE pgrep calls — pgrep's regex alternation with \| does NOT work
# reliably on this system. Each call is its own simple substring match.
is_chain_alive() {
    pgrep -f run_experiment   > /dev/null && return 0
    pgrep -f launch_overnight > /dev/null && return 0
    pgrep -f launch_calibration > /dev/null && return 0
    return 1
}

latest_chain_log() {
    ls -t "$LOGDIR"/overnight_*.log 2>/dev/null | head -1
}

latest_calibration_log() {
    ls -t "$LOGDIR"/calibration_*.log 2>/dev/null | head -1
}

chain_finished_ok() {
    local log
    log=$(latest_chain_log)
    [ -n "$log" ] && grep -q "ALL DONE" "$log"
}

calibration_finished_ok() {
    local log
    log=$(latest_calibration_log)
    [ -n "$log" ] && grep -q "CALIBRATION DONE" "$log"
}

restart_chain() {
    local count
    count=$(cat "$RESTART_COUNT_FILE")
    count=$((count + 1))
    echo "$count" > "$RESTART_COUNT_FILE"
    heartbeat "RESTART #${count}/${MAX_RESTARTS} of main chain"
    if [ "$count" -gt "$MAX_RESTARTS" ]; then
        heartbeat "ABORT — restart cap hit, see ${CRASH}"
        echo "Restart cap ${MAX_RESTARTS} hit at $(date -Is)" > "$CRASH"
        tail -100 "$(latest_chain_log)" >> "$CRASH" 2>/dev/null
        return 1
    fi
    cd "$ROOT" && bash launch_overnight.sh
    return 0
}

launch_calibration_once() {
    if [ -f "$CALIBRATION_LAUNCHED" ]; then return; fi
    touch "$CALIBRATION_LAUNCHED"
    heartbeat "Main chain DONE — launching calibration (A1 + A2)"
    cd "$ROOT" && bash launch_calibration.sh
}

while true; do
    if [ -f "$FINAL_DONE" ]; then
        heartbeat "FINAL_DONE present — watchdog exiting"
        write_status "FINAL_DONE — everything is finished, safe to read results"
        exit 0
    fi
    if [ -f "$CRASH" ]; then
        heartbeat "CRASH file present — watchdog exiting"
        write_status "CRASH — manual intervention needed, see ${CRASH}"
        exit 1
    fi

    if is_chain_alive; then
        # Reset the dead-checks counter on every confirmed alive sample
        echo 0 > "$DEAD_CHECKS_FILE"
        heartbeat "alive"
    else
        # Increment dead-checks counter — only restart after N consecutive dead checks
        dead_checks=$(cat "$DEAD_CHECKS_FILE")
        dead_checks=$((dead_checks + 1))
        echo "$dead_checks" > "$DEAD_CHECKS_FILE"

        # Could be a transient pgrep miss or pod-busy stutter. Be patient.
        heartbeat "no process detected (dead-check $dead_checks/$DEAD_CHECKS_THRESHOLD)"

        if [ "$dead_checks" -ge "$DEAD_CHECKS_THRESHOLD" ]; then
            # Confirmed dead. Reset the dead-check counter and act.
            echo 0 > "$DEAD_CHECKS_FILE"
            if calibration_finished_ok; then
                touch "$FINAL_DONE"
                heartbeat "Calibration DONE — full pipeline complete"
            elif chain_finished_ok; then
                launch_calibration_once
            else
                if ! restart_chain; then
                    continue
                fi
            fi
        fi
    fi

    sleep "$SLEEP_SEC"
done
