Here we are going to write the neural network which prdicts the 3 main parameters 
1. speculative_num_steps - this says how many tokens to generate parallely like the K number
2. speculative_eagle_topk - here in each token in that k line we predict multiple candidates . SO the draft model says these are my top k sample and these goes into the branching tree for the lookahead seqeunce. Now the main model verifies all these topk elements and decides one.
3. speculative_num_draft_tokens - this says we can verify say 10 tokens parallely with main model for adding in K seq....say k=20 we say 10 tokens can be verified paralllely . This depends on the GPU usage



So the flow inside that speculative decoding is that we use the parameters to predict say k tokens while these tokens are predicited by draft model we parallely verify these with the main model and then append them into the K sequence.