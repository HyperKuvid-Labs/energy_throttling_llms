"""Print sweep progress: rows done, recent results, error count."""

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/sweep.jsonl"
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    rows = []

for r in rows[-4:]:
    cfg = f"bs={r['batch_size']} ({r['steps']},{r['topk']},{r['num_draft_tokens']})"
    if r.get("error"):
        print(f"  {cfg} ERROR {r['error'][:60]}")
    else:
        print(f"  {cfg} accept={r.get('accept_length')} "
              f"speed={r.get('speed_tok_s')} J/tok={r.get('joules_per_token')}")

print(f"errors: {sum(1 for r in rows if r.get('error'))}/{len(rows)}")
