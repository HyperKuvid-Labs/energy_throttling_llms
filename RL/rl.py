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
        profiled_metrics = [np.random.uniform(50.0, 80.0), np.random.uniform(50.0, 80.0), np.random.uniform(60.0, 90.0), 
                            np.random.uniform(3000.0, 6000.0), np.random.uniform(7000.0, 12000.0), np.random.uniform(400000.0, 700000.0), 
                            np.random.uniform(2000.0, 4000.0), np.random.uniform(40.0, 70.0)]
        agent.prolfiled_metrics = profiled_metrics

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

