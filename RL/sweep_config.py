"""Speculative decoding parameter grid for the EAGLE3 sweep.

The three parameters the RL policy controls are not freely combinable. The
constraints below are taken from sglang 0.5.2 source, not from the docs:

  1. steps * topk + 1 >= num_draft_tokens
     scripts/playground/bench_speculative.py skips configs that violate this --
     the surplus draft tokens can never be filled. The matching assert in
     srt/server_args.py is commented out, so the server launches them anyway.

  2. topk == 1 forces num_draft_tokens = steps + 1
     srt/server_args.py rewrites the value and only logs a warning. Two of the
     three actions collapse into one, so these must be deduplicated or the
     critic learns from an action that was never executed.

  3. (steps, topk, num_draft_tokens) == (0, 0, 0) is the special-cased
     non-speculative baseline. Any other zero-containing combination is invalid.
"""

BATCH_SIZES = [1, 4, 8, 16]
STEPS = [0, 1, 3, 5]
TOPK = [0, 1, 2, 4]
NUM_DRAFT_TOKENS = [0, 2, 4, 8, 16]

# Reference speculative config used for the Phase 0 headline number. The SGLang
# docs show (3, 4, 16), but 3*4+1 = 13 < 16 violates constraint 1 -- their own
# bench script would skip it. (3, 4, 8) is the nearest valid config.
REFERENCE_CONFIG = (3, 4, 8)
BASELINE_CONFIG = (0, 0, 0)


def canonicalize(steps, topk, num_draft_tokens):
    """Apply the server-side rewrite so a config maps to what actually runs."""
    if steps > 0 and topk == 1:
        num_draft_tokens = steps + 1
    return steps, topk, num_draft_tokens


def is_valid(steps, topk, num_draft_tokens):
    if (steps, topk, num_draft_tokens) == BASELINE_CONFIG:
        return True
    if steps == 0 or topk == 0 or num_draft_tokens == 0:
        return False
    return steps * topk + 1 >= num_draft_tokens


def build_grid(batch_sizes=None, steps=None, topk=None, num_draft_tokens=None):
    """Deduplicated list of (batch_size, steps, topk, num_draft_tokens)."""
    batch_sizes = BATCH_SIZES if batch_sizes is None else batch_sizes
    steps = STEPS if steps is None else steps
    topk = TOPK if topk is None else topk
    num_draft_tokens = NUM_DRAFT_TOKENS if num_draft_tokens is None else num_draft_tokens

    seen = set()
    grid = []
    for bs in batch_sizes:
        for s in steps:
            for k in topk:
                for d in num_draft_tokens:
                    if not is_valid(s, k, d):
                        continue
                    config = (bs,) + canonicalize(s, k, d)
                    if config in seen:
                        continue
                    seen.add(config)
                    grid.append(config)
    return grid


if __name__ == "__main__":
    grid = build_grid()
    print(f"{len(grid)} configs")
    for bs in BATCH_SIZES:
        rows = [c for c in grid if c[0] == bs]
        print(f"  bs={bs}: {len(rows)}")
        for _, s, k, d in rows:
            tag = "  <- baseline" if (s, k, d) == BASELINE_CONFIG else ""
            tag = "  <- reference" if (s, k, d) == REFERENCE_CONFIG else tag
            print(f"    steps={s:<2} topk={k:<2} draft={d:<2}{tag}")
