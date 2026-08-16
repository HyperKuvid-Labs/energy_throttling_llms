#!/usr/bin/env bash
# Run the full 38-config sweep on the pod, resuming wherever it left off.
#
# The pod is a spot instance and can be preempted mid-sweep. run_sweep.py
# appends one JSONL row per config, so the resume index is just the number of
# rows already written -- rerunning this script after a preemption picks up at
# the next config instead of repeating work already paid for.
set -euo pipefail

OUT=${OUT:-/root/sweep.jsonl}
LOG=${LOG:-/root/sweep.log}

START=0
if [ -f "$OUT" ]; then
    START=$(wc -l < "$OUT")
fi

echo "resuming at index $START -> $OUT"
cd /root
export SGLANG_RECORD_STEP_TIME=1
nohup python3 run_sweep.py --output "$OUT" --start "$START" >> "$LOG" 2>&1 &
echo "sweep PID $!"
