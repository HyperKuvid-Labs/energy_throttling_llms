#!/usr/bin/env bash
# Serialize on the single GPU: finish the in-flight bs=16 retry, retry the
# remaining bs=8/steps=5 failures at the same lower mem-fraction, then run a
# 3rd full-grid repeat pass (indices 152-227) so every action is seen 3x.
set -uo pipefail
cd /home/pradheep/Documents/energy_throttling_llms/RL
export SGLANG_RECORD_STEP_TIME=1

echo "=== waiting for bs16 retry (PID 120273) to finish ==="
while kill -0 120273 2>/dev/null; do sleep 5; done
echo "=== bs16 retry done ==="

echo "=== retrying bs=8 failures at mem-fraction 0.55 ==="
python3 retry_bs16.py --source laptop_sweep.jsonl --output bs8_retry.jsonl \
    --batch-size 8 --mem-fraction 0.55
echo "=== bs8 retry done ==="

echo "=== running repeat=2 full grid pass at mem-fraction 0.55 ==="
python3 run_sweep.py --target unsloth/Llama-3.2-1B-Instruct \
    --draft rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct \
    --output laptop_sweep.jsonl --port 30001 --mem-fraction 0.55 \
    --repeats 3 --start 152
echo "=== chain complete ==="
