# Components

The idea is to use the following architecture:

**Two Actors:**
- Target actor
- Fast updating actor

**One Critic:**
- Q-target critic

## How It Works

The process works as follows:

- The policy outputs a state with three parameters. If the state is within some boundary, we proceed.
- Calculate the Q-value of this state, but don't update the policy yet.
- When the next state comes, we take the TD error and then update the policy.
- After some iterations, the batch updates the target policy. Until now, only the target policy gives out the value for Q. We'll be using the target policy for Q-value estimation while updating the fast updating policy. This may be confusing in description, but when I code it up, it's easier to understand.

## Profiling

This project also contains a file for CPU and GPU metric profiling during function execution, on which the reward is based. It's available in the file [profiler_cpu_gpu.py](profiler_cpu_gpu.py).
