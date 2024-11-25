#!/usr/bin/env python3

# # agentemu.py -- DRL-SSxApp Emulator for Training an DQN with Captured Data from 3 Network Slices

# Deep Q-Network (DQN) is a reinforcement learning algorithm that combines
# Q-learning with deep neural networks to approximate the action-value function.
# The key idea is to use a deep neural network to estimate the Q-values, which
# represent the expected cumulative reward obtained from taking a certain action
# in a given state and following an optimal policy thereafter.


# ## Algorithm Overview
#
# 1.  Initialize the Q-network with random weights and set the target network
#     with the same weights.
# 2.  For each episode:
#     - Observe the current state ( s ).
#     - Select an action ( a ) using an (\\epsilon)-greedy policy.
#     - Execute the action ( a ), receive the reward ( r ), and observe the next
#       state ( s' ).
#     - Store the transition ( (s, a, r, s') ) in the replay buffer.
#     - Sample a mini-batch of transitions from the replay buffer.
#     - Compute the target value ( y ) and update the Q-network using gradient
#       descent.
#     - Update the target network periodically.




# ## **Code**

# ### **Imports**
import torch
import signal
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import os

# For the get_state function
from common import get_state, perform_action

# ### **Hyperparameters and Constants**

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# | Parameter              | Default Value | Description                                                                  | Updated Value |
# | ---------------------- | ------------- | ---------------------------------------------------------------------------- | ------------- |
# | `LR`                   | `3e-4`        | Learning Rate: Controls the step size during model weight updates.           | `1e-4`        |
# | `BATCH_SIZE`           | `128`         | Number of experiences to sample from the replay buffer for each training batch. | `32`          |
# | `BUFFER_SIZE`          | `1e5`         | Size of the replay buffer that stores past experiences for learning.         | `1e5`         |
# | `UPDATE_EVERY`         | `8`           | Frequency of model parameter updates during training.                        | `4`           |
# | `TAU`                  | `1e-3`        | Soft update parameter for target network updates.                            | `5e-4`        |
# | `EPISODE_MAX_TIMESTEP` | `300`         | Maximum number of timesteps per episode.                                     | `3`           |
# | `eps_start`            | `1.0`         | Initial value of epsilon for exploration vs. exploitation in action selection. | `1.0`         |
# | `eps_end`              | `0.01`        | Final value of epsilon after decay.                                          | `0.01`        |
# | `eps_decay`            | `0.995`       | Rate at which epsilon decays over time.                                      | `0.995`       |
# | `state_len`            | `8`           | Dimensionality of the state space.                                           | `3`           |
# | `action_len`           | `8`           | Dimensionality of the action space.                                          | `4`           |
# | `n_episodes`           | `1500`        | Number of episodes for training the agent.                                   | `300000`        |
# | `max_t`                | `300`         | Maximum number of timesteps per episode.                                     | `3`           |





# ### **Hyperparameters and environment-specific constants**
LR = 1e-4  # Slightly higher learning rate for faster convergence
BATCH_SIZE = 32  # Larger batch size for more efficient updates
BUFFER_SIZE = int(1e5)  # Keep buffer size the same if memory allows
UPDATE_EVERY = 4  # More frequent updates
TAU = 5e-4  # Faster soft updatesng the target network. Determines how much of the local model's weights are copied to the target network.

# Constants for PRB and DL Bytes mapping and thresholds
PRB_INC_RATE = 6877         # Determines how many Physical Resource Blocks (PRB) to increase when adjusting PRB allocation.
# We expect 27,253,551
DL_BYTES_THRESHOLD = [19922669, 6670690, 660192] # Thresholds for different slice types: Embb, Medium, Urllc.

REWARD_PER_EPISODE = []  # List to store rewards for each episode
EPISODE_MAX_TIMESTEP = 3  # Maximum number of timesteps per episodes


        # Initialize a counter for unique episodes
global unique_episode_counter







# ### **Q-Network**
#
# This class definition implements a Q-Network using PyTorch, which is a type of
# neural network used in reinforcement learning to approximate the Q-value
# function.


# ### Q-Learning
#
# Q-learning is a value-based reinforcement learning algorithm that aims to
# learn the optimal action-value function, ( Q^\*(s, a) ), which satisfies the
# Bellman equation.

# ### Experience Replay
#
# Experience replay is a mechanism used to store the agent's experiences, ( (s,
# a, r, s') ), in a replay buffer. During training, mini-batches of experiences
# are sampled uniformly from this buffer. This approach helps to break the
# correlation between consecutive experiences, making the training process more
# stable.

# ### Target Network
#
# To stabilize the learning process, DQN uses a separate target network to
# compute the target Q-values. The parameters of the target network are updated
# periodically to match the parameters of the main Q-network, which reduces
# oscillations and divergence.

# ### Loss Function
#
# The DQN algorithm minimizes a loss function defined as:

# $$ L(\\theta) = \\mathbb{E}\_{(s, a, r, s') \\sim \\text{ReplayBuffer}}
# \\left\[ \\left( y - Q(s, a; \\theta) \\right)^2 \\right\], $$

# where $$ ( y = r + \\gamma \\max\_{a'} Q(s', a'; \\theta^-) ), ( \\theta ) $$
# are the parameters of the Q-network, and $$ ( \\theta^- ) $$ are the
# parameters of the target network.


# **init**: Initializes the Q-Network with the given parameters and defines the
# layers of the network with ReLU activations. forward: Performs a forward pass
# through the network, taking an input state and returning the predicted
# Q-values for each action.
class DQN_QNetwork(nn.Module):
    """
    Q-Network for approximating the Q-value function.
    The Q-network predicts Q-values for each action given a state.
    """
    def __init__(self, state_len, action_len, seed, layer1_size=128, layer2_size=128, layer3_size=128, layer4_size=128):
        super(DQN_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Define the layers of the Q-network with ReLU activations
        #
        # ReLU (Rectified Linear Unit) is applied to the output of each layer in
        # the neural network. Its mathematical form is: \\\[ f(x) = \\max(0, x)
        # \\\] Where: <span class="hljs-bullet">-</span> \\( x \\) is the input
        # to the activation function. <span class="hljs-bullet">-</span> The
        # output is \\( x \\) if \\( x > 0 \\), otherwise, it is \\( 0 \\).
        self.l1 = nn.Linear(state_len, layer1_size)
        self.l2 = nn.Linear(layer1_size, layer2_size)
        self.l3 = nn.Linear(layer2_size, layer3_size)
        self.l4 = nn.Linear(layer3_size, layer4_size)
        self.l5 = nn.Linear(layer4_size, action_len)

    def forward(self, input_state):
        # Forward pass through the network
        x = F.relu(self.l1(input_state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

 # ### **Agent**
 #
 # The Agent class represents an agent that interacts with an environment and
 # learns from it using a Q-learning approach.


# **init**: Initializes the agent with state and action lengths, seed, and DDQN
# flag. step: Adds an experience to the replay buffer and updates the model.
# act: Chooses an action using an epsilon-greedy policy. learn: Updates the
# Q-network based on sampled experiences. soft_update: Updates the target
# Q-network by interpolating between local and target models.




class DQN():
    """
    Agent that interacts with the environment and learns from it using a Q-learning approach.
    """
    def __init__(self, state_len, action_len, seed, DDQN=False):
        self.action_len = action_len
        self.state_len = state_len
        self.seed = random.seed(seed)

        # Initialize the local and target Q-networks
        self.qnetwork_local = DQN_QNetwork(state_len, action_len, seed).to(device)
        self.qnetwork_target = DQN_QNetwork(state_len, action_len, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = DQN_ReplayBuffer(action_len, BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0
        self.DDQN = DDQN


# A mini-batch is a small subset of size (B) sampled uniformly at random from
# the replay buffer ( \\mathcal{D} ). Let ( {(s_i, a_i, r_i, s_i')}_{i=1}^B )
# represent the sampled mini-batch. This means:

# $$
# (s_i, a_i, r_i, s_i') \sim \text{Uniform}(\mathcal{D})
# $$

# **Purpose**:

# - Ensures the samples are i.i.d. (independent and identically distributed)**, reducing the risk of overfitting to temporally correlated experiences.
# - Enables efficient stochastic gradient descent (SGD) updates.
    def step(self, state, action, reward, next_state, done):
        # Add experience to replay buffer and update the model if necessary
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, 0.99)

    def act(self, state, eps=0.):
# Choose an action using an epsilon-greedy policy
#
# The \\(\\epsilon\\)-greedy policy is used to balance exploration and
# exploitation during training. The action \\( a \\) is selected as
# follows:
#
# $$
# a = 
# \begin{cases} 
# \text{random action} & \text{with probability } \epsilon, \\
# \arg\max_a Q(s, a; \theta) & \text{with probability } 1 - \epsilon.
# \end{cases}
# $$
#
# Where:
#
# - \\(\\epsilon\\) is the exploration rate, decaying over time (e.g., \\(\\epsilon\\) decreases exponentially during training).
# - \( Q(s, a; \theta) \) is the Q-value predicted by the neural network for state \( s \) and action \( a \).

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_len))

    def learn(self, experiences, gamma):
        # Update the Q-network based on the sampled experiences
        states, actions, rewards, next_states, dones = experiences

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        Q_targets_next = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# ### **Replay Buffer**
#
# the DQN_ReplayBuffer class is used to store and sample experiences, which are
# used in reinforcement learning to train an agent

# init: Initializes the buffer with the specified action size, buffer size, and
# batch size. It also creates a named tuple called Experience to store the
# state, action, reward, next state, and done information. add: Adds a new
# experience to the buffer. The experience is represented by a named tuple.
# sample: Samples a batch of experiences from the buffer. It randomly selects
# batch_size number of experiences and converts them into torch tensors. len:
# Returns the number of experiences in the buffer.

class DQN_ReplayBuffer:
    """
    Replay Buffer for storing and sampling experiences.
    """
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.random()

    def add(self, state, action, reward, next_state, done):
        # Add experience to the buffer
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # Sample a batch of experiences from the buffer
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# ### **Training Function**
#
# This is a Deep Q-Network (DQN) training function in Python. It trains an agent
# to make decisions in an environment with the goal of maximizing a reward
# signal. The function takes in several parameters that control the training
# process, such as the number of episodes, maximum timesteps, and exploration
# rate.


# Initializes the agent, environment, and training parameters. Loops through
# episodes, with each episode consisting of a sequence of timesteps. At each
# timestep, the agent chooses an action based on the current state and
# exploration rate. The environment responds with a reward and next state. The
# agent updates its experience buffer with the current state, action, reward,
# and next state. The agent updates its Q-network using the experience buffer.
# The exploration rate is decayed over time to encourage exploitation. The
# training process is repeated for a specified number of episodes. The trained
# model is saved to a file.

def run_dqn(agent, n_episodes=1500, max_t=3, eps_start=1.0, eps_end=0.01, eps_decay=0.995, pth_file='checkpoint.pth', malicious_chance=1000, malicious_chance_increase=0.0):
    """
    Train the DQN agent with specified parameters and save checkpoints.
    """

    unique_episode_counter = 0

    eps = eps_start  # Initialize epsilon (exploration rate)

    actions = [] 
    rewards = []
    total_prbs = [[],[],[]]
    dl_bytes = [[],[],[]]
  # action_prbs = \[2897, 965, 96\]
    action_prbs = [2897, 965, 91]   # eMBB, Medium, URLL
    total = 0

    total = 0
    correct = 0
    reward_averages = list()
    percentages = list()
    action_count = [0 for x in range(4)]
    EPISODE_MAX_TIMESTEP = max_t

    for episode in range(1, n_episodes + 1):
        done = False
        temp = [-1] * EPISODE_MAX_TIMESTEP
        max_t = 0
        i = 1
        # assigned PRBs to each slice
       # action_prbs = \[2897, 965, 96\]
        action_prbs = [2897, 965, 91]  # eMBB, Medium, URLLC

        global DL_BYTE_TO_PRB_RATES
        # number of DL bytes that each slice increases by per PRB
        DL_BYTE_TO_PRB_RATES = [6877, 6877, 6877]

    
        next_state = get_state(action_prbs, DL_BYTE_TO_PRB_RATES, malicious_chance)  # Get the initial state from the dataframes
        while not done and max_t < EPISODE_MAX_TIMESTEP:
            total += 1
            state = next_state
            action = agent.act(np.array(state), eps)  # Choose an action based on the current state
            action_count[action] += 1

            for i, prb_value in enumerate(total_prbs):
                total_prbs[i].append(prb_value)
                dl_bytes[i].append(state[i])

            reward, done, action_prbs = perform_action(action, state, i, action_prbs, DL_BYTES_THRESHOLD)
            actions.append(action)
            rewards.append(reward)
            if reward > 0:
                correct += 1
            malicious_chance += malicious_chance_increase
            next_state = get_state(action_prbs, DL_BYTE_TO_PRB_RATES, malicious_chance)  # Get the initial state from the dataframes
            agent.step(state, action, reward, next_state, done)  # Update the agent with the experience

            # Update the score with the reward
            max_t += 1  # Increment the timestep
            i += 1  # Increment the step counter

        if episode % 400 == 0:  # Decay epsilon every 500 episodes
            eps = max(eps_end, eps_decay * eps)
            
        # Store the score and actions
        actions.append(temp)


        # Update the state lists with the current state





            # Increment the unique episode counter
        unique_episode_counter += 1

            # Check if 100 unique episodes have been processed
        if unique_episode_counter >= 1000:
            avg_reward = sum(rewards[-1000:]) / 1000  # Average of the last 100 rewards

            print(f'\rEpisode {episode}\tAverage Score: {avg_reward:.2f}', end="")
            reward_averages.append(avg_reward)
            percentages.append(correct / total)

                # Reset the unique episode counter
            unique_episode_counter = 0
        if episode % 300000 == 0:
            print(f'\rEpisode {episode}\treward: {avg_reward}')
            print("Percentage: ", correct / total)
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))

            ax[0].plot(reward_averages, 'r')
            ax[0].set_title('Reward')

            ax[1].plot(percentages, color='g')
            ax[1].set_title('Percentages')

            ax[2].bar(range(len(action_count)), action_count, color='b')
            ax[2].set_title('Actions taken')

            action_count = [0 for x in range(4)]

            plt.show()


    # Save the model checkpoint
    torch.save(agent.qnetwork_local.state_dict(), pth_file)
    print(f'Model saved to {pth_file}')

    return rewards, (correct, total)
                





