"""Launch an SGLang EAGLE3 server, run one prompt through it, and tear it down.

Renamed from sglang.py: that filename shadowed the installed `sglang` package,
so `from sglang.test.doc_patch import ...` resolved to this file and failed with
"No module named 'sglang.test'; 'sglang' is not a package".

For dataset collection use run_sweep.py instead -- it reuses one server across a
whole batch and records throughput/energy. This module is a single-shot helper.
"""

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process
import openai

# Ungated mirror of meta-llama/Llama-3.1-8B-Instruct (identical config: hidden
# 4096, vocab 128256). The gated meta-llama repo needs manual access approval.
DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

# EAGLE3 draft heads are trained against one specific target model -- this is not
# a free choice. Must pair with the Llama-3.1-8B target above.
DEFAULT_DRAFT_MODEL = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"


def eagle_3_sd(speculative_num_steps, speculative_eagle_topk, speculative_num_draft_tokens,
               model=DEFAULT_MODEL, draft_model=DEFAULT_DRAFT_MODEL,
               prompt="What's happening in thinkingmachines.ai", max_tokens=64):
    # topk > 1 with page_size > 1 silently produces wrong results on paged
    # attention backends other than flashinfer (srt/server_args.py).
    attention_backend = "--attention-backend flashinfer" if speculative_eagle_topk > 1 else ""

    server_process, port = launch_server_cmd(
    f"""
    python3 -m sglang.launch_server \
        --model-path {model} \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path {draft_model} \
        --speculative-num-steps {speculative_num_steps} \
        --speculative-eagle-topk {speculative_eagle_topk} \
        --speculative-num-draft-tokens {speculative_num_draft_tokens} \
        --mem-fraction-static 0.85 \
        --cuda-graph-max-bs 2 \
        {attention_backend} \
        --log-level warning
    """
    )

    #Here the port variable is chosen randomly by the launch_server_cmd function because
    #while we run that launch_server_cmd it binds to some random tcp server and gives the port

    try:
        wait_for_server(f"http://localhost:{port}")
        client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )

        print_highlight(f"Response: {response}")
        print_highlight(f"The actual resp: {response.choices[0].message.content}")

        return response.choices[0].message.content
    finally:
        # Without this every call leaks a server process holding the GPU.
        terminate_process(server_process)


def main():
    prompt = "Some shot over here, doesn't matter as of now"
    # Reference config: satisfies steps*topk + 1 >= num_draft_tokens.
    speculative_num_steps = 3
    speculative_eagle_topk = 4
    speculative_num_draft_tokens = 8
    resp = eagle_3_sd(speculative_num_steps, speculative_eagle_topk, speculative_num_draft_tokens,
                      prompt=prompt)
    print(resp)


if __name__ == "__main__":
    main()
