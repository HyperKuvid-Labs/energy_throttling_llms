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
speculative_    num_steps = 5
speculative_eagle_topk = 8
speculative_num_draft_tokens = 32
#Eagle3 decoding

server_process, port = launch_server_cmd(
f"""
python3 -m sglang.launch_server \
    --model google/gemma-2b-it \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path google/gemma-1.1b-it \
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

# print_highlight(f"Response: {response}")