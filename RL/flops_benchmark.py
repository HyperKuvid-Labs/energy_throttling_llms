"""Estimate decode FLOPs for the three configurations used in this repo.

This is an algorithmic dense-linear FLOPs estimate, not a CUDA utilization
counter. One multiply-add counts as two FLOPs. The estimate includes the
target transformer and vocabulary head, plus EAGLE3's fusion layer, one-layer
draft transformer, and 32k draft vocabulary head. Attention score/value
matmuls, norms, activations, sampling, and tree-building kernels are excluded.

The default acceptance lengths come from the full quality run, so speculative
cycles are converted to FLOPs per output token using measurements from the
same workload. Pass --results to recompute from another quality-results file.

Usage:
  python3 flops_benchmark.py
  python3 flops_benchmark.py --results quality_results_full.json \
      --output flops_results.json
"""

import argparse
import json
from pathlib import Path

from transformers import AutoConfig


TARGET_MODEL = "unsloth/Llama-3.2-1B-Instruct"
DRAFT_MODEL = "rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct"

CONFIGS = [
    ("no_spec", (0, 0, 0), "no_spec"),
    ("chosen_3_2_4", (3, 2, 4), "chosen_bs16"),
    ("chosen_3_4_8", (3, 4, 8), "chosen_bs1_4_8"),
]

DEFAULT_ACCEPTANCE_LENGTHS = {
    "no_spec": 1.0,
    "chosen_3_2_4": 2.129,
    "chosen_3_4_8": 2.425,
}


def llama_linear_flops(config):
    """Dense linear FLOPs for one target-model decode token."""
    hidden = config.hidden_size
    head_dim = getattr(config, "head_dim", hidden // config.num_attention_heads)
    query = config.num_attention_heads * head_dim
    key_value = config.num_key_value_heads * head_dim

    attention_weights = hidden * (query + 2 * key_value) + query * hidden
    mlp_weights = 3 * hidden * config.intermediate_size
    transformer_weights = config.num_hidden_layers * (attention_weights + mlp_weights)
    vocabulary_head_weights = hidden * config.vocab_size

    return {
        "transformer": 2 * transformer_weights,
        "vocabulary_head": 2 * vocabulary_head_weights,
        "total": 2 * (transformer_weights + vocabulary_head_weights),
    }


def eagle3_linear_flops(config):
    """Dense linear FLOPs for the one-layer EAGLE3 draft model."""
    if config.num_hidden_layers != 1:
        raise ValueError("this estimator expects the SGLang one-layer EAGLE3 architecture")

    hidden = config.hidden_size
    head_dim = getattr(config, "head_dim", hidden // config.num_attention_heads)
    query = config.num_attention_heads * head_dim
    key_value = config.num_key_value_heads * head_dim

    # SGLang's Llama EAGLE3 layer concatenates token embeddings and draft
    # hidden states before QKV, so the QKV input width is 2 * hidden.
    attention_weights = (2 * hidden) * (query + 2 * key_value) + query * hidden
    mlp_weights = 3 * hidden * config.intermediate_size
    core_weights = attention_weights + mlp_weights

    # Target hidden states from three captured layers are concatenated, then
    # projected from 3 * hidden to hidden during draft-cache extension.
    fusion_weights = 3 * hidden * hidden
    draft_vocab_size = config.draft_vocab_size
    vocabulary_head_weights = hidden * draft_vocab_size

    return {
        "core": 2 * core_weights,
        "target_hidden_fusion": 2 * fusion_weights,
        "vocabulary_head": 2 * vocabulary_head_weights,
    }


def estimate_config(config, acceptance_length, target, draft, output_tokens):
    steps, topk, num_draft_tokens = config
    if steps == 0:
        cycle_flops = target["total"]
        output_flops = cycle_flops
        target_tokens = 1.0
        draft_extend_tokens = 0.0
        draft_expand_tokens = 0.0
    else:
        # Each speculative cycle verifies num_draft_tokens candidates with the
        # target. EAGLE3 then extends its cache over the accepted output tokens
        # and performs (steps - 1) tree-expansion forwards of width topk.
        target_tokens = float(num_draft_tokens)
        draft_extend_tokens = acceptance_length
        draft_expand_tokens = float(topk * (steps - 1))

        target_flops = target_tokens * target["total"]
        draft_extend_flops = (
            draft_extend_tokens * (draft["core"] + draft["target_hidden_fusion"])
            + draft["vocabulary_head"]
        )
        draft_expand_flops = draft_expand_tokens * (
            draft["core"] + draft["vocabulary_head"]
        )
        cycle_flops = target_flops + draft_extend_flops + draft_expand_flops
        output_flops = cycle_flops / acceptance_length

    return {
        "config": list(config),
        "acceptance_length": acceptance_length,
        "target_tokens_per_cycle": target_tokens,
        "draft_extend_tokens_per_cycle": draft_extend_tokens,
        "draft_expand_tokens_per_cycle": draft_expand_tokens,
        "estimated_dense_linear_gflops_per_cycle": cycle_flops / 1e9,
        "estimated_dense_linear_gflops_per_output_token": output_flops / 1e9,
        "estimated_dense_linear_tflops_per_request": output_flops * output_tokens / 1e12,
    }


def read_acceptance_lengths(path):
    with path.open() as f:
        data = json.load(f)

    lengths = {}
    for name, _config, result_key in CONFIGS:
        try:
            lengths[name] = float(data[result_key]["perf"]["accept_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing acceptance length for {result_key!r} in {path}") from exc
    return lengths


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET_MODEL)
    parser.add_argument("--draft", default=DRAFT_MODEL)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, default=here / "flops_results.json")
    parser.add_argument("--output-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.output_tokens <= 0:
        parser.error("--output-tokens must be positive")

    target_config = AutoConfig.from_pretrained(args.target, local_files_only=True)
    draft_config = AutoConfig.from_pretrained(args.draft, local_files_only=True)
    target = llama_linear_flops(target_config)
    draft = eagle3_linear_flops(draft_config)
    if args.results is None:
        acceptance_lengths = DEFAULT_ACCEPTANCE_LENGTHS
        acceptance_length_source = "full quality run reported in README.md"
    else:
        acceptance_lengths = read_acceptance_lengths(args.results)
        acceptance_length_source = args.results.name

    rows = []
    for name, config, _result_key in CONFIGS:
        row = estimate_config(
            config,
            acceptance_lengths[name],
            target,
            draft,
            args.output_tokens,
        )
        row["name"] = name
        rows.append(row)

    baseline = rows[0]["estimated_dense_linear_gflops_per_output_token"]
    reference = rows[2]["estimated_dense_linear_gflops_per_output_token"]
    for row in rows:
        per_token = row["estimated_dense_linear_gflops_per_output_token"]
        row["flops_ratio_vs_no_spec"] = per_token / baseline
        row["flops_reduction_vs_3_4_8_pct"] = (1 - per_token / reference) * 100

    report = {
        "method": "algorithmic dense-linear FLOPs; multiply-add counts as 2 FLOPs",
        "speculative_cycle_formula": "D*T + A*(C+F) + H + K*(S-1)*(C+H)",
        "formula_terms": {
            "D": "num_draft_tokens verified by the target per cycle",
            "T": "target FLOPs per verified token",
            "A": "measured accepted output tokens per cycle",
            "C": "EAGLE3 transformer-core FLOPs per token",
            "F": "EAGLE3 target-hidden-state fusion FLOPs per extended token",
            "H": "EAGLE3 vocabulary-head FLOPs per evaluation",
            "K": "speculative_eagle_topk",
            "S": "speculative_num_steps",
        },
        "excluded": [
            "attention score/value matmuls",
            "normalization and activation elementwise operations",
            "sampling, top-k, and speculative tree kernels",
            "prompt prefill",
        ],
        "acceptance_length_source": acceptance_length_source,
        "target_model": args.target,
        "draft_model": args.draft,
        "output_tokens_per_request": args.output_tokens,
        "component_gflops": {
            "target_per_verified_token": target["total"] / 1e9,
            "draft_core_per_token": draft["core"] / 1e9,
            "draft_target_hidden_fusion_per_extended_token": draft["target_hidden_fusion"] / 1e9,
            "draft_vocabulary_head_per_evaluation": draft["vocabulary_head"] / 1e9,
        },
        "results": rows,
    }

    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"{'configuration':<18} {'accept':>8} {'GFLOPs/output tok':>18} "
        f"{'TFLOPs/512 tok':>16} {'vs no-spec':>11}"
    )
    for row in rows:
        print(
            f"{row['name']:<18} {row['acceptance_length']:>8.3f} "
            f"{row['estimated_dense_linear_gflops_per_output_token']:>18.3f} "
            f"{row['estimated_dense_linear_tflops_per_request']:>16.3f} "
            f"{row['flops_ratio_vs_no_spec']:>10.2f}x"
        )
    reduction = rows[1]["flops_reduction_vs_3_4_8_pct"]
    print(f"\n(3,2,4) uses {reduction:.1f}% fewer estimated FLOPs than (3,4,8).")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
