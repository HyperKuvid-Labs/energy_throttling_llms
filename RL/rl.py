import torch
import torch.nn.functional as F

from components.target_actor import TargetActorNetwork
from components.fast_updating_actor import FastActorNetwork
from components.q_value import QNetwork

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # minimum usage of X server resources
import time

from sglang import eagle_3_sd
from components.profiler_cpu_gpu import profiler

prompts = [
    # ========== MATHEMATICAL REASONING (AIME/MATH Level) ==========
    "Find the number of positive integers n ≤ 1000 for which there exists a positive real number x such that n = x⌊x⌋. Show all steps in your reasoning.",

    "Let p be the least prime number for which there exists a positive integer n such that n^4 + 1 is divisible by p^2. Find the least positive integer m such that m^4 + 1 is divisible by p^2. Provide detailed step-by-step solution.",

    "Compute the value of: 1/(2×3) + 1/(3×4) + 1/(4×5) + ... + 1/(99×100) + 1/(100×101). Express your answer as a simplified fraction and explain each step.",

    "Let ABCD be a convex quadrilateral with AB = CD = 10, BC = 14, and AD = 2√65. Assume that the diagonals of ABCD intersect at point P, and that the sum of the areas of triangles APB and CPD equals the sum of the areas of triangles BPC and APD. Find the area of quadrilateral ABCD.",

    "Determine all polynomials P(x) with real coefficients such that P(x^2 + x + 1) divides P(x^3 - 1) for all real numbers x. Justify your reasoning thoroughly.",

    "Find the smallest positive integer n such that if n^2 is written in base 10, its digits sum to exactly 2025. Show your complete calculation process.",

    "Let f(x) = x^3 - 3x + 1. Find all real solutions to f(f(f(x))) = 0. Provide a complete analytical solution with verification.",

    "A sequence {a_n} satisfies a_1 = 1, a_2 = 2, and a_{n+2} = a_{n+1} + a_n for n ≥ 1. Find the largest integer k such that the product a_1 × a_2 × ... × a_k is less than 10^100. Explain your approach.",

    "Prove that for any positive integer n, the number 2^(2^n) + 1 has at least n distinct prime factors. Provide a rigorous mathematical proof.",

    "Let S be the set of all positive integers n for which 1/n has a terminating decimal expansion. Find the sum of all elements in S that are less than or equal to 1000.",

    "Solve the system of equations: x^3 + y^3 + z^3 = 3xyz, x + y + z = 3, x^2 + y^2 + z^2 = 5. Find all real solutions with complete justification.",

    "Find the number of ordered triples (a,b,c) of positive integers satisfying lcm(a,b) = 1000, lcm(b,c) = 2000, and lcm(c,a) = 2000.",

    "Let f: ℝ → ℝ be a continuous function satisfying f(x+y) = f(x) + f(y) + xy(x+y) for all x,y ∈ ℝ and f(1) = 1. Find f(2025).",

    "Determine the number of ways to tile a 3×20 rectangle using 1×3 tiles. Derive a recurrence relation and solve it.",

    "Find all functions f: ℕ → ℕ such that f(f(n)) + f(n+1) = n + 2 for all positive integers n. Prove uniqueness if applicable.",

    # ========== ADVANCED CODING CHALLENGES ==========
    "Implement a CUDA kernel for sparse matrix-vector multiplication (SpMV) using the CSR format with shared memory optimization, warp-level primitives, and proper coalescing. Include detailed comments explaining memory access patterns, bank conflict avoidance, and performance considerations. Provide both the kernel code and host code.",

    "Write a complete implementation of the Aho-Corasick algorithm for multiple pattern matching in C++. Include construction of the failure function, goto function, and output function. Optimize for cache locality and include comprehensive test cases with time complexity analysis.",

    "Implement a lock-free concurrent hash map in C++ using atomic operations and compare-and-swap primitives. Handle ABA problems, memory reclamation with hazard pointers, and provide benchmarks comparing it to std::unordered_map with mutex locks.",

    "Create a production-ready implementation of Monte Carlo Tree Search for Connect-4 with the following requirements: UCB1 selection, rapid action value estimation, virtual loss for parallel MCTS, neural network integration for policy and value functions, and configurable exploration parameters. Include complete training loop.",

    "Implement a B-tree database index structure in Rust with support for range queries, bulk loading, concurrent access with MVCC, and write-ahead logging for crash recovery. Include comprehensive unit tests and performance benchmarks.",

    "Write a complete ray tracer in C++ with support for: recursive ray tracing, multiple light sources, shadow rays, reflection and refraction, anti-aliasing with stratified sampling, BVH acceleration structure, and parallel rendering. Include detailed comments on mathematical formulations.",

    "Implement the Earley parser algorithm for context-free grammars with support for ambiguous grammars, left recursion, and epsilon productions. Include parse tree construction and comprehensive test cases for various grammar types.",

    "Create a GPU-accelerated implementation of the Fast Fourier Transform using CUDA with Cooley-Tukey algorithm, optimized butterfly operations, shared memory tiling, and bit-reversal permutation. Compare performance with CPU implementation.",

    "Implement a complete LR(1) parser generator that takes a context-free grammar and produces parsing tables. Handle shift-reduce and reduce-reduce conflicts with appropriate precedence rules. Include error recovery mechanisms.",

    "Write a production-ready implementation of the Raft consensus algorithm with: leader election, log replication, safety guarantees, snapshotting, cluster membership changes, and comprehensive failure scenario testing.",

    "Implement a differentiable renderer using PyTorch with support for: mesh rendering, texture mapping, Phong shading, soft rasterization, and gradient backpropagation through the rendering pipeline for inverse graphics tasks.",

    "Create a complete implementation of the A* pathfinding algorithm with: admissible heuristics, tie-breaking strategies, jump point search optimization for grid maps, dynamic obstacle handling, and visualization of explored nodes.",

    "Implement a custom memory allocator in C with: segregated free lists, coalescing of adjacent free blocks, boundary tags, alignment handling, and thread-local caches for improved performance in multi-threaded applications.",

    "Write a complete compiler frontend for a subset of C including: lexical analysis with regex-based tokenization, recursive descent parsing, abstract syntax tree construction, symbol table management, type checking, and semantic analysis with detailed error reporting.",

    "Implement a distributed MapReduce framework in Go with: job scheduling, fault tolerance through re-execution, data locality awareness, combine functions for optimization, and support for multiple reduce tasks running in parallel.",

    # ========== CHAIN-OF-THOUGHT REASONING ==========
    "You are debugging a CUDA kernel that shows 50% lower performance than expected. The kernel performs matrix multiplication with tiling. Walk through a systematic debugging process: Step 1: Identify potential bottlenecks (memory bandwidth, compute, latency). Step 2: Use CUDA profiling tools (nvprof, Nsight Compute) to measure metrics. Step 3: Analyze memory access patterns for coalescing. Step 4: Check for bank conflicts in shared memory. Step 5: Verify occupancy and register usage. Step 6: Implement and test optimizations. Provide detailed analysis for each step with specific metrics to check.",

    "A binary search tree becomes unbalanced after many insertions. Design a complete solution: Step 1: Explain why unbalanced BSTs degrade performance. Step 2: Compare self-balancing alternatives (AVL, Red-Black, B-trees). Step 3: Choose the most appropriate structure with justification. Step 4: Implement the rebalancing algorithm with code. Step 5: Analyze time complexity for all operations. Step 6: Provide test cases demonstrating balanced behavior.",

    "You need to optimize a deep learning model that's too large for GPU memory. Develop a comprehensive strategy: Step 1: Profile memory usage by layer and identify memory bottlenecks. Step 2: Evaluate gradient checkpointing trade-offs. Step 3: Consider mixed precision training (FP16/BF16). Step 4: Analyze model parallelism vs data parallelism. Step 5: Implement zero-redundancy optimizer (ZeRO). Step 6: Benchmark throughput and convergence. Provide detailed implementation for each technique.",

    "Design a distributed caching system for a high-traffic web application. Reasoning process: Step 1: Define requirements (latency, consistency, availability). Step 2: Choose cache eviction policy (LRU, LFU, ARC) with justification. Step 3: Design cache invalidation strategy. Step 4: Implement consistent hashing for distribution. Step 5: Handle cache stampede scenarios. Step 6: Add monitoring and metrics. Step 7: Provide complete architecture diagram and code.",

    "Analyze why a particular machine learning model exhibits high variance. Systematic approach: Step 1: Visualize learning curves (training vs validation loss). Step 2: Calculate bias-variance decomposition. Step 3: Identify overfitting indicators. Step 4: Evaluate regularization techniques (L1, L2, dropout, early stopping). Step 5: Consider data augmentation strategies. Step 6: Implement cross-validation for robust evaluation. Step 7: Provide empirical results with multiple techniques.",

    # ========== SYSTEM DESIGN & ARCHITECTURE ==========
    "Design a real-time collaborative code editor like Google Docs for code. Address: conflict resolution with Operational Transformation or CRDT, WebSocket communication architecture, cursor position synchronization, syntax highlighting across clients, undo/redo in collaborative context, presence awareness, offline mode with sync, scalability to 100+ concurrent editors. Provide detailed system architecture, data structures, and protocol specifications.",

    "Architect a distributed message queue system similar to Apache Kafka. Include: partitioning strategy for throughput, replication for fault tolerance, exactly-once delivery semantics, consumer group management, offset tracking, log compaction, zero-copy transfers, back-pressure handling. Provide complete design document with pseudo-code for critical components.",

    "Design a recommendation system for an e-commerce platform with 100M users and 10M products. Cover: collaborative filtering vs content-based approaches, matrix factorization techniques, cold start problem solutions, real-time personalization pipeline, A/B testing framework, feature engineering for deep learning models, handling implicit feedback, scalability architecture with offline batch processing and online serving.",

    "Create a comprehensive design for a distributed file system like HDFS. Include: block storage and replication, namenode architecture and metadata management, datanode operations, rack awareness for replica placement, fault tolerance and recovery mechanisms, consistency model, read/write protocols, balancing and decommissioning. Provide detailed protocol descriptions.",

    "Design an API rate limiting system that can handle 1M requests per second. Address: token bucket vs leaky bucket algorithms, distributed rate limiting with Redis, handling clock skew, sliding window implementation, per-user and per-endpoint limits, graceful degradation, monitoring and alerting. Include complete implementation with data structures and algorithms.",

    "Architect a video streaming platform like YouTube. Cover: video ingestion and transcoding pipeline, adaptive bitrate streaming (HLS/DASH), CDN integration and edge caching, recommendation algorithm, view count tracking at scale, comment system with moderation, live streaming infrastructure, cost optimization strategies. Provide microservices architecture diagram.",

    "Design a distributed transaction system supporting ACID properties across multiple databases. Include: two-phase commit protocol, handling coordinator failures, timeouts and deadlock detection, isolation levels implementation, distributed locking mechanisms, compensation for long-running transactions. Provide complete protocol specification with failure scenarios.",

    "Create a comprehensive monitoring and alerting system for microservices architecture. Address: metrics collection (push vs pull), distributed tracing with OpenTelemetry, log aggregation and analysis, anomaly detection algorithms, alert correlation and noise reduction, SLA tracking, root cause analysis automation. Include data pipeline architecture.",

    "Design a geo-distributed database system ensuring low latency worldwide. Cover: multi-region replication strategies, conflict resolution in eventual consistency, read/write splitting, data sharding by geography, consistency models (strong, causal, eventual), failover procedures, cost optimization. Provide detailed replication protocol.",

    "Architect a real-time analytics platform processing 1TB of data per hour. Include: stream processing with Apache Flink/Spark Streaming, exactly-once processing guarantees, windowing strategies, late data handling, state management at scale, queryable state, integration with batch processing (Lambda architecture), OLAP database selection. Provide complete pipeline architecture.",

    # ========== REINFORCEMENT LEARNING & AI ==========
    "Implement a complete Proximal Policy Optimization (PPO) algorithm from scratch in PyTorch for continuous control tasks. Include: actor-critic architecture, advantage estimation with GAE, clipped surrogate objective, value function loss, entropy regularization, learning rate scheduling, gradient clipping, parallel environment collection with multiprocessing, and comprehensive training loop. Provide detailed mathematical derivations for each component and explain hyperparameter choices.",

    "Design and implement a Deep Q-Network (DQN) with all modern improvements: double Q-learning to reduce overestimation, dueling architecture separating value and advantage streams, prioritized experience replay with importance sampling, multi-step returns, distributional RL with C51, noisy networks for exploration. Include complete code with replay buffer implementation and training metrics visualization.",

    "Create a sophisticated reward shaping framework for reinforcement learning in sparse reward environments. Address: potential-based reward shaping maintaining optimal policy, curiosity-driven exploration with intrinsic motivation, hindsight experience replay for goal-conditioned tasks, reward prediction for auxiliary tasks, curriculum learning strategy. Implement in a challenging environment like Montezuma's Revenge.",

    "Implement AlphaZero from scratch for the game of Chess. Include: neural network architecture for policy and value heads, Monte Carlo Tree Search with PUCT selection, self-play data generation pipeline, training loop with experience replay, move selection with temperature parameter, resignation threshold, evaluation against baseline agents. Provide complete training infrastructure.",

    "Design a multi-agent reinforcement learning system for cooperative tasks. Cover: centralized training with decentralized execution (CTDE), value decomposition networks (VDN, QMIX), communication protocols between agents, credit assignment problem, emergent coordination behaviors, opponent modeling. Implement for a multi-agent particle environment.",

    "Create an offline reinforcement learning algorithm based on Conservative Q-Learning (CQL). Address: overestimation in offline setting, pessimistic value estimation, behavior regularization, importance sampling for off-policy correction, uncertainty quantification. Include complete implementation with benchmark on D4RL datasets and comparison with behavioral cloning.",

    "Implement a model-based reinforcement learning approach combining Dyna-style planning with learned world models. Include: transition model learning with neural networks, reward model, planning with learned model, handling model errors and uncertainty, trading off real vs simulated experience. Provide complete implementation for a continuous control task.",

    "Design a hierarchical reinforcement learning system using the Options framework. Cover: option discovery algorithms (clustering, skill chaining), intra-option learning, hierarchical value functions, temporal abstraction benefits, transfer learning across tasks. Implement for a complex navigation task with multiple sub-goals.",

    "Create a safe reinforcement learning system with constrained optimization. Address: Constrained Policy Optimization (CPO), safety constraints formulation, Lagrangian relaxation, primal-dual methods, worst-case performance guarantees, recovery policies. Implement for a safety-critical robotic control task.",

    "Implement inverse reinforcement learning to recover reward functions from expert demonstrations. Include: maximum entropy IRL, apprenticeship learning, generative adversarial imitation learning (GAIL), comparison with behavioral cloning, handling suboptimal demonstrations. Provide complete implementation with evaluation metrics.",

    # ========== TECHNICAL EXPLANATIONS (High Redundancy) ==========
    "Explain the complete compilation process of a C++ program from source code to executable binary. Step 1: Preprocessing - macro expansion, file inclusion, conditional compilation. Step 2: Lexical analysis - tokenization, handling comments and whitespace. Step 3: Syntax analysis - parsing, AST construction, grammar rules. Step 4: Semantic analysis - type checking, symbol resolution, scope rules. Step 5: Intermediate representation - three-address code, SSA form. Step 6: Optimization - constant folding, dead code elimination, loop optimization, inline expansion. Step 7: Code generation - instruction selection, register allocation. Step 8: Assembly - machine code generation. Step 9: Linking - symbol resolution, relocation, static vs dynamic linking. Step 10: Loading - executable format (ELF/PE), memory layout. Provide detailed examples for each step.",

    "Provide a comprehensive explanation of how modern CPUs achieve high performance through pipelining and out-of-order execution. Part 1: Basic pipeline stages - fetch, decode, execute, memory, writeback. Part 2: Pipeline hazards - structural, data, control hazards and their solutions. Part 3: Branch prediction - static vs dynamic, two-level adaptive predictors, tournament predictors. Part 4: Speculative execution - register renaming, reorder buffer, reservation stations. Part 5: Memory hierarchy - cache levels, cache coherence protocols (MESI), false sharing. Part 6: SIMD instructions - vectorization, AVX/SSE. Part 7: Performance analysis - CPI breakdown, bottleneck identification. Include specific examples from x86-64 architecture.",

    "Explain the complete lifecycle of a TCP connection with packet-level details. Step 1: Three-way handshake - SYN, SYN-ACK, ACK packets, initial sequence numbers. Step 2: Data transmission - sliding window protocol, acknowledgments, cumulative ACK. Step 3: Flow control - receive window, window scaling. Step 4: Congestion control - slow start, congestion avoidance, fast retransmit, fast recovery, CUBIC algorithm. Step 5: Error detection - checksum calculation, segment validation. Step 6: Retransmission - timeout calculation, exponential backoff. Step 7: Connection termination - FIN packets, TIME_WAIT state. Step 8: Socket API - system calls, kernel buffers. Provide Wireshark-level packet analysis examples.",

    "Describe the complete process of rendering a frame in a modern game engine. Phase 1: Application stage - game logic update, physics simulation, animation. Phase 2: Culling - view frustum culling, occlusion culling, portal-based culling. Phase 3: Scene graph traversal - spatial data structures, level of detail selection. Phase 4: Geometry processing - vertex transformation, skinning, morphing. Phase 5: Rasterization - triangle setup, fragment generation, early depth test. Phase 6: Fragment shading - texture sampling, lighting calculations, normal mapping. Phase 7: Post-processing - HDR tone mapping, bloom, motion blur, anti-aliasing (MSAA, TAA, FXAA). Phase 8: Display - v-sync, frame buffering, tear prevention. Provide OpenGL/Vulkan API calls for each phase.",

    "Explain how Git implements version control internally with object model and data structures. Part 1: Object types - blob (file content), tree (directory structure), commit (snapshot), tag. Part 2: Object storage - SHA-1 hashing, content-addressable storage, object database. Part 3: References - branches as pointers, HEAD, detached HEAD state. Part 4: Merge strategies - three-way merge, recursive strategy, conflict resolution. Part 5: Rebase - rewriting history, interactive rebase, force pushing. Part 6: Remote operations - fetch, pull, push protocols. Part 7: Efficiency - delta compression, pack files, garbage collection. Provide low-level plumbing commands demonstrating each concept.",

    # ========== ALGORITHMS WITH DETAILED PROOFS ==========
    "Prove the correctness of Dijkstra's shortest path algorithm using the greedy choice property and optimal substructure. Step 1: State the algorithm clearly with pseudocode. Step 2: Define the invariant maintained by the algorithm. Step 3: Prove the greedy choice property - locally optimal choice leads to global optimum. Step 4: Prove optimal substructure - optimal solution contains optimal subsolutions. Step 5: Prove termination and time complexity analysis. Step 6: Discuss why negative edges break the algorithm. Step 7: Provide a complete worked example on a weighted graph.",

    "Provide a rigorous proof that the RSA cryptosystem is correct and analyze its security. Part 1: Mathematical foundations - Euler's totient function, Fermat's Little Theorem, Chinese Remainder Theorem. Part 2: Key generation algorithm - prime selection, modulus calculation, public/private exponent derivation. Part 3: Correctness proof - show encryption followed by decryption recovers plaintext using modular arithmetic. Part 4: Security analysis - hardness of integer factorization, chosen plaintext attacks, timing attacks. Part 5: Practical considerations - padding schemes (OAEP), key size recommendations. Part 6: Complete implementation with number-theoretic primitives.",

    "Prove that the Bellman-Ford algorithm correctly computes shortest paths even with negative edge weights and detects negative cycles. Step 1: Algorithm description with clear pseudocode. Step 2: State the relaxation property. Step 3: Prove correctness by induction on number of edges in shortest path. Step 4: Prove negative cycle detection mechanism. Step 5: Time complexity analysis - O(VE) justification. Step 6: Compare with Dijkstra's algorithm and explain when to use each. Step 7: Worked example with negative edges.",

    "Prove the correctness of the KMP string matching algorithm and derive its time complexity. Part 1: Naive algorithm and its inefficiency. Part 2: Failure function construction - computing longest proper prefix that is also suffix. Part 3: Matching phase using failure function. Part 4: Correctness proof - show no matches are missed. Part 5: Time complexity analysis - amortized analysis showing O(n+m). Part 6: Comparison with Boyer-Moore and Rabin-Karp algorithms. Part 7: Complete implementation with detailed comments.",

    "Prove that the Ford-Fulkerson algorithm computes maximum flow correctly and analyze its termination. Step 1: Flow network definitions - capacity constraints, flow conservation. Step 2: Residual graph construction. Step 3: Augmenting path theorem. Step 4: Max-flow min-cut theorem statement and proof. Step 5: Algorithm correctness proof. Step 6: Termination analysis - rational capacities guarantee. Step 7: Edmonds-Karp improvement with BFS. Step 8: Worked example with residual graph evolution.",

    # ========== LONG-FORM TECHNICAL WRITING ==========
    "Write a comprehensive tutorial on implementing a production-ready REST API using FastAPI with PostgreSQL database. Cover: project structure and dependencies, database models with SQLAlchemy ORM, Alembic migrations, CRUD operations with async/await, request validation with Pydantic, authentication using JWT tokens, authorization with role-based access control, error handling and custom exceptions, API documentation with OpenAPI, logging and monitoring, rate limiting middleware, CORS configuration, Docker containerization, deployment to cloud platforms, performance optimization with connection pooling, caching strategies with Redis, testing with pytest including fixtures and mocks, CI/CD pipeline with GitHub Actions. Provide complete code examples for each section with explanations.",

    "Create a detailed guide for optimizing Python code for performance. Section 1: Profiling - cProfile, line_profiler, memory_profiler usage and interpretation. Section 2: Built-in optimizations - list comprehensions vs loops, generator expressions, set operations. Section 3: NumPy vectorization - broadcasting, avoiding loops, efficient array operations. Section 4: Cython - writing C extensions, type annotations, compilation. Section 5: Numba - JIT compilation, parallel execution, CUDA support. Section 6: Multiprocessing vs threading - GIL implications, process pools. Section 7: Async I/O - asyncio, aiohttp, when to use. Section 8: Memory optimization - slots, interning, weak references. Section 9: Algorithmic improvements - data structure selection. Section 10: Benchmarking methodology - timeit, comparative analysis. Provide before/after code examples with performance metrics.",

    "Write an in-depth explanation of how neural networks learn through backpropagation. Chapter 1: Perceptron foundations - linear separability, activation functions. Chapter 2: Multilayer networks - universal approximation theorem. Chapter 3: Forward pass - matrix operations, layer computations. Chapter 4: Loss functions - MSE, cross-entropy, their derivatives. Chapter 5: Backpropagation algorithm - chain rule application, gradient computation. Chapter 6: Computational graphs - automatic differentiation. Chapter 7: Optimization algorithms - SGD, momentum, Adam, learning rate schedules. Chapter 8: Initialization strategies - Xavier, He, impact on training. Chapter 9: Regularization techniques - L2, dropout, batch normalization. Chapter 10: Vanishing/exploding gradients - ReLU, residual connections. Chapter 11: Complete implementation from scratch in NumPy. Provide mathematical derivations with matrix notation.",

    "Develop a comprehensive guide to concurrent programming in C++ with modern features. Part 1: Threading basics - std::thread, join vs detach, thread lifecycle. Part 2: Mutual exclusion - std::mutex, lock_guard, unique_lock, shared_mutex. Part 3: Condition variables - wait, notify, spurious wakeups. Part 4: Atomics - memory orders, compare_exchange, lock-free programming. Part 5: Memory model - happens-before relationship, synchronization points. Part 6: Future and promises - async tasks, packaged_task. Part 7: Thread pools - work stealing, task queues. Part 8: Lock-free data structures - queues, stacks. Part 9: Parallel algorithms - execution policies. Part 10: Common pitfalls - deadlocks, race conditions, debugging techniques. Provide complete working examples for each concept.",

    "Write a detailed explanation of the Linux kernel's process scheduler. Section 1: Process states - running, ready, waiting, zombie. Section 2: Scheduling classes - CFS, real-time, deadline. Section 3: CFS algorithm - virtual runtime, red-black tree, time slices. Section 4: Priority and nice values - calculation, inheritance. Section 5: Multiprocessor scheduling - load balancing, CPU affinity. Section 6: Real-time scheduling - SCHED_FIFO, SCHED_RR, priority inversion. Section 7: Context switching - overhead, optimization. Section 8: Completely Fair Queueing - group scheduling, cgroups. Section 9: Performance analysis - /proc statistics, perf tool. Section 10: Kernel configuration - tuning parameters. Provide code references from actual Linux kernel source.",

    # ========== CODE REVIEW & DEBUGGING SCENARIOS ==========
    "Review this CUDA kernel for matrix multiplication and identify all performance issues:\n``````\nAnalyze: 1) Memory access patterns and coalescing, 2) Shared memory utilization, 3) Bank conflicts, 4) Occupancy and register usage, 5) Comparison with cuBLAS performance. Then provide an optimized version using tiling with detailed explanations.",

    "Debug this Python code that's causing a memory leak in a long-running web server:\n``````\nExplain: 1) Why this causes a memory leak, 2) How Python's garbage collection works, 3) Reference counting vs garbage collection, 4) Weak references solution, 5) LRU cache with size limits, 6) Memory profiling with tracemalloc. Provide fixed implementations with multiple approaches.",

    "Identify the race condition in this concurrent code:\n``````\nExplain: 1) Non-atomic read-modify-write sequence, 2) Thread interleaving scenarios, 3) Lost update problem, 4) Solutions using mutex, atomic operations, 5) Memory ordering guarantees, 6) Performance comparison. Provide corrected implementations with benchmarks.",

    "Review this SQL query for performance optimization:\n``````\nAnalyze: 1) Query execution plan with EXPLAIN, 2) Missing indexes, 3) Join order optimization, 4) WHERE vs HAVING for filtering, 5) Covering indexes, 6) Query rewrite for better performance. Provide optimized version with index suggestions.",

    "Debug why this React component is causing unnecessary re-renders:\n``````\nExplain: 1) Array creation on every render, 2) Inline function creation, 3) useMemo for expensive computations, 4) useCallback for event handlers, 5) React.memo for component memoization, 6) Profiling with React DevTools. Provide optimized version with performance comparison.",
]

q_values = []
rewards = []
actor_losses = []
critic_losses = []

class Agent():
    def __init__(self, input_dims, state_dims, prolfiled_metrics, baseline_metrics, epsilon=1.0, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.prolfiled_metrics = prolfiled_metrics
        self.baseline_metrics = baseline_metrics
        self.epsilon = epsilon

        self.epsilon_min = 0.00001
        self.epsilon_decay = 1e-6

        self.target_actor = TargetActorNetwork(input_dims)
        self.fast_actor = FastActorNetwork(input_dims)
        self.critic = QNetwork(state_dims)

        self.target_actor.load_state_dict(self.fast_actor.state_dict())

    def choose_action(self, observation):
        # epsion greedy action selection
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.fast_actor.device)
            actions = self.fast_actor.forward(state)
            action = actions.cpu().detach().numpy()[0]
        else:
            action = np.random.randint([1, 1, 1], [32, 10, 64])
        return action

    def update_network_parameters(self):
        for target_param, param in zip(self.target_actor.parameters(), self.fast_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def calculate_reward(self, prolfiled_metrics, profiling_baseline_metrics):
        # so the reward is the cumulative sum of metrics[i]/baseline_metrics[i] for i in range(len(metrics))/len(metrics)
        reward = 0
        for i in range(len(prolfiled_metrics)):
            reward += prolfiled_metrics[i]/profiling_baseline_metrics[i]
        reward = reward/len(prolfiled_metrics)
        return reward

    def critic_loss(self, state, action, reward, baseline_metrics):
        # so in ths i need to store s,a,r in the q_values
        self.fast_actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        if len(q_values) < 1:
            print("No previous Q values, returning 0 loss")
            return 0.0
        else:
            q_value_now = self.critic.forward(state, action)
            q_value_old = q_values[-1][0]
            reward = self.calculate_reward(self.prolfiled_metrics, self.baseline_metrics)
            target = reward + self.gamma * q_value_old
            target = torch.tensor(target, dtype=torch.float).to(self.critic.device)
            critic_loss = F.mse_loss(q_value_now, target)
            q_values.append([q_value_now.cpu().detach().numpy(), state.cpu().detach().numpy(), action.cpu().detach().numpy(), reward])
            return critic_loss

    def actor_loss(self, state):
        self.fast_actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        actions = self.fast_actor.forward(state)
        actor_loss = -self.critic.forward(state, actions)
        actor_loss = torch.mean(actor_loss)
        return actor_loss

    def update_fast_actor_critic(self, actor_loss, critic_loss):
        self.fast_actor.update_weights(actor_loss)
        self.critic.update_weights(critic_loss)
        self.update_network_parameters()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self):
        if len(q_values) < 1:
            print("No Q values to learn from")
            return

        state = torch.tensor(q_values[-1][1], dtype=torch.float).to(self.critic.device)
        action = torch.tensor(q_values[-1][2], dtype=torch.float).to(self.critic.device)

        critic_loss = self.critic_loss(state, action, q_values[-1][3], self.baseline_metrics)
        if isinstance(critic_loss, float) and critic_loss == 0.0:
            return
        actor_loss = self.actor_loss(state)

        self.update_fast_actor_critic(actor_loss, critic_loss)
        self.decrement_epsilon()

        print(f"Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Epsilon: {self.epsilon}")

        # q_values.clear()

    def save_models(self):
        print("saving models -> fast_actor, target_actor, critic")
        torch.save(self.fast_actor.state_dict(), 'fast_actor.pth')
        torch.save(self.target_actor.state_dict(), 'target_actor.pth')
        torch.save(self.critic.state_dict(), 'critic.pth')
        print("saved!!")

    def load_models(self):
        try:
            self.fast_actor.load_state_dict(torch.load('fast_actor.pth'))
            self.target_actor.load_state_dict(torch.load('target_actor.pth'))
            self.critic.load_state_dict(torch.load('critic.pth'))
            print("loaded models -> fast_actor, target_actor, critic")
        except Exception as e:
            print(f"Error: {e}")

def plot_learning_curve():
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('learning_curve.png')
    plt.close()

def plot_actor_critic_loss_curve():
    plt.plot(actor_losses, label='Actor Loss')
    plt.plot(critic_losses, label='Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('actor_critic_loss_curve.png')
    plt.close()

def main():
    # should do the profiling here, and also set the baseline metrics too
    # for now, simple
    input_dims = 8 # the metrics like how much ram is avaiable
    state_dims = 8 # the profiling metrics
    available_metrics = [8.0, 7.0, 9.0, 6000.0, 12000.0, 700000.0, 4000.0, 70.0]
    profiled_metrics = [70.0, 65.0, 80.0, 4000.0, 8000.0, 500000.0, 3000.0, 50.0]
    baseline_metrics = [60.0, 60.0, 70.0, 5000.0, 10000.0, 600000.0, 3500.0, 60.0]

    agent = Agent(input_dims=input_dims, state_dims=state_dims, prolfiled_metrics=profiled_metrics, baseline_metrics=baseline_metrics)

    n_iterations = 10000

    for i in range(n_iterations):
        state = available_metrics
        action = agent.choose_action(state)
        print(f"Chosen Action: {action}")

        # should do the profiling here with the chosen action
        # for now, just random profiled metrics
        x = np.random.uniform(0, 76)
        prompt = prompts[int(x)]
        print(f"Prompt: {prompt}...")

        profiler(eagle_3_sd)(action[0], action[1], action[2], prompt)

        # profiled_metrics = [np.random.uniform(50.0, 80.0), np.random.uniform(50.0, 80.0), np.random.uniform(60.0, 90.0),
        #                     np.random.uniform(3000.0, 6000.0), np.random.uniform(7000.0, 12000.0), np.random.uniform(400000.0, 700000.0),
        #                     np.random.uniform(2000.0, 4000.0), np.random.uniform(40.0, 70.0)]
        # agent.prolfiled_metrics = profiled_metrics

        # here i need to notedown the format, and get the avg of all the metrics for the time of excution,         for all the cpu and gpu metrics notes, and need to also fill the baseline metrics accordingly.
        # Will do it as soon as possible !!

        reward = agent.calculate_reward(profiled_metrics, baseline_metrics)
        rewards.append(reward)
        print(f"Reward: {reward}")

        agent.learn()
        actor_losses.append(agent.actor_loss(torch.tensor([state], dtype=torch.float).to(agent.fast_actor.device)).item())
        critic_losses.append(agent.critic_loss(torch.tensor([state], dtype=torch.float).to(agent.critic.device),
                                               torch.tensor([action], dtype=torch.float).to(agent.critic.device),
                                               reward, baseline_metrics).item())

        if i % 100 == 0 and i > 0:
            plot_learning_curve()
            plot_actor_critic_loss_curve()
            print(f"Saved learning curves at iteration {i}")\

            time.sleep(1) # to ensure the plots are saved properly

        plot_learning_curve()
        time.sleep(0.1)

if __name__ == "__main__":
    main()

