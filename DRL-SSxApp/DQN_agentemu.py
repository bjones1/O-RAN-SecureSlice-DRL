#!/usr/bin/env python3


#


# ## **Objective**
#
# The objective is to maximize the overall throughput for all slices by mapping
# resource blocks (RBs) while ensuring minimum QoS agreements are met and
# maximum thresholds for each slice are upheld via secure slicing:

# $ \\max\_{T_1, T_2, T_3} \\sum\_{i=1}^{3} T_i $

# where $ T_1 $, $ T_2 $, and $ T_3 $ represent the throughput for URLLC, eMBB,
# and Medium slices, respectively.

# ## **Constraints**

# The optimization problem is subject to the following constraints:

# 1.  **QoS Constraint for URLLC:** The throughput of URLLC must satisfy the
#     ultra-reliable and low-latency requirement while being constrained by the
#     minimum QoS threshold for the Medium slice. This can be expressed as:

# $ \\theta\_{\\text{min}} < \\text{QoS}\_{\\text{URLLC}}(T_1) <
# \\theta\_{\\text{URLLC}} $

# where $ \\theta\_{\\text{URLLC}} $ is the minimum QoS threshold for the URLLC
# slice and $ \\theta\_{\\text{min}} $ is the minimum QoS threshold for the
# Medium slice.

# 2.  **QoS Constraint for eMBB:** The throughput of eMBB must meet the enhanced
#     broadband requirement:

# $ \\text{QoS}\_{\\text{eMBB}}(T_2) \\geq \\theta\_{\\text{eMBB}} \\quad
# \\text{(high data rate requirement)} $

# where $ \\theta\_{\\text{eMBB}} $ is the minimum QoS threshold for the eMBB
# slice.

# 3.  **QoS Constraint for Medium Slice:** The throughput of the Medium slice
#     must satisfy its QoS requirements, which is between URLLC and eMBB:

# $ \\theta\_{\\text{URLLC}} \\leq \\text{QoS}\_{\\text{Medium}}(T_3) \\leq
# \\theta\_{\\text{eMBB}} $

# 4.  **Resource Allocation Constraint:** The total allocated resources
#     (Physical Resource Blocks) for all slices cannot exceed the available
#     resources $ R\_{\\text{total}} $:

# $ \\sum\_{i=1}^{3} R_i \\leq R\_{\\text{total}} $

# where $ R_i $ is the resource allocation for slice $ i $.

# 5.  **Secure Slicing Constraint:** All UEs in the slice must operate below the
#     maximum threshold; otherwise, they are considered malicious and may
#     compromise the resources of legitimate UEs. Let $ \\tau\_{\\text{max}} $
#     represent the maximum allowable resource usage per UE. The constraint can
#     be expressed as:

# $ S\_{u,i} < \\tau\_{\\text{max}}, \\quad \\forall u \\in U_i $

# where $ S\_{u,i} $ is the resource usage for UE $ u $ in slice $ i $, and $
# U_i $ is the set of UEs in slice $ i $.

# ## **Approach**

# ![](../documentation/images/drl-ss-xapp-1.png)

# The diagram outlines a framework for allocating physical resource blocks
# (PRBs) among user equipment (UE) slices using a Deep Reinforcement Learning
# (DRL) agent in a secure slicing xApp environment.

# #### Key Components
#
# - **DRL Agent**: Uses Key Performance Metrics (KPM) as input, actions for PRB
#   reallocation, and network throughput as the reward. The agent's operations
#   include:
#   - **State**: Input from KPM.
#   - **Action**: Adjust PRB allocations.
#   - **Reward**: Based on network throughput performance.
# - **Slice Configurator**: Implements the DRL agent’s actions by reallocating
#   PRBs to slices.
# - **Secure Slicing xApp**: Executes resource adjustments and handles UE
#   security.
# - **UE Instances**: Devices bound to slices, with the potential to be moved to
#   a secure slice if detected as malicious.
# - **iperf3 Server**: Generates traffic in the Downlink for user traffic.

# #### DRL Agent Actions
#
# 1 - 3. **Increase PRBs**: Adds 5 PRBs if a slice's throughput is insufficient
# but below or equal to the SLA. Removes 5 PRB if throughput exceeds the SLA
# requirements and gives 0 reward for that episode. 4. **Secure UE**: Moves UEs
# to a secure slice if malicious behavior is detected.

# #### Workflow
#
# 1.  The DRL agent receives real-time KPM data and adjusts PRBs through the
#     Slice Configurator.
# 2.  The iperf3 server provides throughput feedback to guide resource
#     allocation.
# 3.  A reward function evaluates actions to maximize UE throughput while
#     ensuring fairness and preventing malicious activity.

# #### Security & Fairness
#
# - **Proportional Fairness**: Ensures balanced PRB allocation across slices.
# - **Secure Slicing**: Isolates malicious UEs, allocating no PRBs to them, to
#   protect network integrity.

# ## **srsRAN Scheduling Breakdown and System Model**

# ![](../documentation/images/ssxapp.png)

# In srsRAN 4G, the configuration provided maps resource blocks (RBs) based on
# several key parameters. With SISO (Single Input, Single Output) and
# transmission mode (TM1), the system allocates RBs across the physical layer
# considering bandwidth (bw=10), which corresponds to a 10 MHz system, and the
# subframe configuration. The scheduler assigns RBs dynamically based on user
# demands, channel conditions, and available bandwidth. The MCS (modulation and
# coding scheme) is set to automatic, allowing the system to adaptively select
# the most efficient scheme. PDSCH and PUSCH are then mapped to these allocated
# RBs for downlink and uplink transmissions, optimizing the use of the available
# frequency-time resources.

# We have 50 RBs in 10 MHz and 100 in 20 Mhz. We have alot more than this though
# since they are virtualized for the purpose of the DRL-SSxApp.
# https://github.com/openaicellular/srsRAN-e2/blob/master/srsenb/enb.conf.example


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


# | Parameter              | Default Value | Description                                                                  | Try 1 |
# | ---------------------- | ------------- | ---------------------------------------------------------------------------- | ----- |
# | `LR`                   | `3e-4`        | Learning Rate: Controls the step size during model weight updates.           |       |
# | `BATCH_SIZE`           | `128`         | num of experiences to sample from the replay buffer for each training batch. |       |
# | `BUFFER_SIZE`          | `1e5`         | Size of the replay buffer that stores past experiences for learning.         |       |
# | `UPDATE_EVERY`         | `8`           | Frequency of model parameter updates during training.                        |       |
# | `TAU`                  | `1e-3`        | Soft update parameter for target network updates.                            |       |
# | `EPISODE_MAX_TIMESTEP` | `300`         | Maximum number of timesteps per episode.                                     |       |
# | `eps_start`            | `1.0`         | Initial value of epsilon for exploration vs. exploitation in action selectio |       |
# | `eps_end`              | `0.01`        | Final value of epsilon after decay.                                          |       |
# | `eps_decay`            | `0.995`       | Rate at which epsilon decays over time.                                      |       |
# | `state_len`            | `8`           | Dimensionality of the state space.                                           |       |
# | `action_len`           | `8`           | Dimensionality of the action space.                                          |       |
# | `n_episodes`           | `1500`        | Number of episodes for training the agent.                                   |       |
# | `max_t`                | `300`         | Maximum number of timesteps per episode.                                     |       |





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

    def step(self, state, action, reward, next_state, done):
        # Add experience to replay buffer and update the model if necessary
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, 0.99)

    def act(self, state, eps=0.):
        # Choose an action using an epsilon-greedy policy
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
# the DQN_ReplayBuffer class is used to store and sample experiences, which are used
# in reinforcement learning to train an agent

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
                

            

# ### **Create Data Frames**

# This code reads CSV files from three directories (Embb, Medium, Urllc) into
# lists of Pandas data frames, calculates the number of data frames and their
# lengths, and returns the data frames and their lengths.





# ### Results

# DQN:

# # \### **Hyperparameters and environment-specific constants**
#
# LR = 1e-4\
# BATCH_SIZE = 32\
# BUFFER_SIZE = int(1e5)\
# UPDATE_EVERY = 4\
# TAU = 5e-4

# action_prbs = \[2897, 965, 91\] if episode % 400 == 0: # Decay epsilon every
# 400 episodes

# final average reward 26444471


# ![image](../documentation/images/DQN.png)







# DDQN:

# # \### **Hyperparameters and environment-specific constants**
#
# LR = 1e-4\
# BATCH_SIZE = 32\
# BUFFER_SIZE = int(1e5)\
# UPDATE_EVERY = 4\
# TAU = 5e-4

# action_prbs = \[2897, 965, 91\] if episode % 400 == 0: # Decay epsilon every
# 400 episodes

# final average reward 26043123



# ![image](../documentation/images/DDQN.png)


# Dueling DQN


# # \### **Hyperparameters and environment-specific constants**
#
# LR = 1e-4\
# BATCH_SIZE = 32\
# BUFFER_SIZE = int(1e5)\
# UPDATE_EVERY = 4\
# TAU = 5e-4

# action_prbs = \[2897, 965, 91\] if episode % 400 == 0: # Decay epsilon every
# 400 episodes

# final average reward 25155529
