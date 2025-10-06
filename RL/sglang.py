from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process
import openai

#Eagle2 decoding

# server_process, port = launch_server_cmd(
#     """
# python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algorithm EAGLE \
#     --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 3 \
#     --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 --cuda-graph-max-bs 8 --log-level warning
# """
# )

# wait_for_server(f"http://localhost:{port}")


# client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

# response = client.chat.completions.create(
#     model="meta-llama/Llama-2-7b-chat-hf",
#     messages=[
#         {"role": "user", "content": "List 3 countries and their capitals."},
#     ],
#     temperature=0,
#     max_tokens=64,
# )

# print_highlight(f"Response: {response}")

#What i am thinking is that we need to use a bigger model for training as it has more overhaed for forward pass
#but max i can run in my lap is ig gemma2B(or something relative to this)

#I am defining these as dummies for now but you get this from policy network
#Eagle3 decoding

def eagle_3_sd(speculative_num_steps, speculative_eagle_topk, speculative_num_draft_tokens, model="meta-llama/Meta-Llama-3.1-8B-Instruct", draft_model="google/gemma-1.1b-it", prompt="What's happening in thinkingmachines.ai"):
    server_process, port = launch_server_cmd(
    f"""
    python3 -m sglang.launch_server \
        --model {model} \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path {draft_model} \
        --speculative-num-steps {speculative_num_steps} \
        --speculative-eagle-topk {speculative_eagle_topk} \
        --speculative-num-draft-tokens {speculative_num_draft_tokens} \
        --mem-fraction 0.6 \
        --cuda-graph-max-bs 2 \
        --dtype float16 \
        --log-level warning
    """
    )

    #Here the port variable is chosen randomly by the launch_server_cmd function because 
    #while we run that launch_server_cmd it binds to some random tcp server and gives the port

    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=0,
        max_tokens=64,
    )

    print_highlight(f"Response: {response}")

    # this is how the resp comes
    # Response: ChatCompletion(id='b26761b38a64449f9fa5d62aceccc4ed', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=' Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1758358129, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})

    print_highlight(f"The actual resp: {response.choices[0].message.content}")

    return response.choices[0].message.content

def main():
    prompt = "Some shot over here, doesn't matter as of now"
    # dummy variables set already
    speculative_num_steps = 5
    speculative_eagle_topk = 8
    speculative_num_draft_tokens = 32
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    draft_model = "google/gemma-1.1b-it"
    resp = eagle_3_sd(speculative_num_steps, speculative_eagle_topk, speculative_num_draft_tokens, model, draft_model, prompt)
    print(resp)