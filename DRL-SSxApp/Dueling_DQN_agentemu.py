#!/usr/bin/env python3

# # agentemu.py -- DRL-SSxApp Emulator for Training an DQN_Dueling with Captured Data from 3 Network Slices


# ## **Code**

# ### **Imports**
import torch
import signal
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import os

from common import get_state, perform_action

# ### **Hyperparameters and Constants**

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# | Parameter             | Default Value | Description                                                                 | Try 1  |
# |-----------------------|---------------|-----------------------------------------------------------------------------|--------| 
# | `LR`                  | `3e-4`        | Learning Rate: Controls the step size during model weight updates.          | 
# | `BATCH_SIZE`          | `128`         | num of experiences to sample from the replay buffer for each training batch.|
# | `BUFFER_SIZE`         | `1e5`         | Size of the replay buffer that stores past experiences for learning.        |
# | `UPDATE_EVERY`        | `8`           | Frequency of model parameter updates during training.                       |
# | `TAU`                 | `1e-3`        | Soft update parameter for target network updates.                           |
# | `EPISODE_MAX_TIMESTEP`| `300`         | Maximum number of timesteps per episode.                                    |
# | `eps_start`           | `1.0`         | Initial value of epsilon for exploration vs. exploitation in action selectio|
# | `eps_end`             | `0.01`        | Final value of epsilon after decay.                                         |
# | `eps_decay`           | `0.995`       | Rate at which epsilon decays over time.                                     |
# | `state_len`           | `8`           | Dimensionality of the state space.                                          |
# | `action_len`          | `8`           | Dimensionality of the action space.                                         |
# | `n_episodes`          | `1500`        | Number of episodes for training the agent.                                  |
# | `max_t`               | `300`         | Maximum number of timesteps per episode.                                    |





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
# This class definition implements a Q-Network using PyTorch, which is a type of neural network used in reinforcement learning to approximate the Q-value function.


# __init__: Initializes the Q-Network with the given parameters and defines the layers of the network with ReLU activations.
# forward: Performs a forward pass through the network, taking an input state and returning the predicted Q-values for each action.
# Updated Q-Network for Dueling DQN
class Dueling_QNetwork(nn.Module):
    """
    Dueling Q-Network for approximating the Q-value function.
    This network has separate streams for state-value and advantage functions.
    """
    def __init__(self, state_len, action_len, seed, layer1_size=128, layer2_size=128):
        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Common feature extraction layers
        self.l1 = nn.Linear(state_len, layer1_size)
        self.l2 = nn.Linear(layer1_size, layer2_size)

        # Separate streams for value and advantage
        self.value_stream = nn.Sequential(
            nn.Linear(layer2_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Outputs a single value
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(layer2_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_len)  # Outputs advantages for each action
        )

    def forward(self, input_state):
        # Forward pass through the common layers
        x = F.relu(self.l1(input_state))
        x = F.relu(self.l2(x))

        # Forward pass through the value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean())
        return q_values


 # ### **DQN_Dueling**
 # 
 # The DQN_Dueling class represents an agent that interacts with an environment and learns from it using a Q-learning approach.


# __init__: Initializes the agent with state and action lengths, seed, and DDQN flag.
# step: Adds an experience to the replay buffer and updates the model.
# act: Chooses an action using an epsilon-greedy policy.
# learn: Updates the Q-network based on sampled experiences.
# soft_update: Updates the target Q-network by interpolating between local and target models.


class DQN_Dueling():
    """
    DQN_Dueling that interacts with the environment and learns from it using a Q-learning approach.
    """
    def __init__(self, state_len, action_len, seed, DDQN=True):
        self.action_len = action_len
        self.state_len = state_len
        self.seed = random.seed(seed)

        # Initialize the local and target Q-networks
        self.qnetwork_local = Dueling_QNetwork(state_len, action_len, seed).to(device)
        self.qnetwork_target = Dueling_QNetwork(state_len, action_len, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = Dueling_ReplayBuffer(action_len, BUFFER_SIZE, BATCH_SIZE)
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

        # Get the Q-values for the current state using the local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Double DQN logic:
        # 1. Use the local model to select the best action in the next state
        next_action = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        # 2. Use the target model to calculate the Q-value for the selected action
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_action)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Perform soft update of the target model
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# ### **Replay Buffer**
# the Dueling_ReplayBuffer class is used to store and sample experiences, which are used in reinforcement learning to train an agent

# init: Initializes the buffer with the specified action size, buffer size, and batch size. It also creates a named tuple called Experience to store the state, action, reward, next state, and done information.
# add: Adds a new experience to the buffer. The experience is represented by a named tuple.
# sample: Samples a batch of experiences from the buffer. It randomly selects batch_size number of experiences and converts them into torch tensors.
# len: Returns the number of experiences in the buffer.

class Dueling_ReplayBuffer:
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
# This is a Deep Q-Network (DQN) training function in Python. It trains an agent to make decisions in an environment with the goal of maximizing a reward signal. The function takes in several parameters that control the training process, such as the number of episodes, maximum timesteps, and exploration rate.


# Initializes the agent, environment, and training parameters.
# Loops through episodes, with each episode consisting of a sequence of timesteps.
# At each timestep, the agent chooses an action based on the current state and exploration rate.
# The environment responds with a reward and next state.
# The agent updates its experience buffer with the current state, action, reward, and next state.
# The agent updates its Q-network using the experience buffer.
# The exploration rate is decayed over time to encourage exploitation.
# The training process is repeated for a specified number of episodes.
# The trained model is saved to a file.

def run_dueling(agent, n_episodes=60000, max_t=4, eps_start=1.0, eps_end=0.01, eps_decay=0.995, pth_file='checkpoint.pth', malicious_chance=1000, malicious_chance_increase=0.0):
    """
    Train the DQN agent with specified parameters and save checkpoints.
    """

    unique_episode_counter = 0

    eps = eps_start  # Initialize epsilon (exploration rate)
    df, df_file_len = create_df()  # Create the dataframes for each slice and get their lengths

    actions = [] 
    rewards = []
    total_prbs = [[],[],[]]
    dl_bytes = [[],[],[]]
  #  action_prbs = [2897, 965, 96]
    action_prbs = [2897, 965, 91]  # eMBB, Medium, URLL make the connection between this and 100 RB for 20 Mhz in LTE.
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
       # action_prbs = [2897, 965, 96]
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

        if episode % 100 == 0:  # Decay epsilon every 500 episodes
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
        if episode % 60000 == 0:
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

# This code reads CSV files from three directories (Embb, Medium, Urllc) into lists of Pandas data frames, calculates the number of data frames and their lengths, and returns the data frames and their lengths.
def create_df():  # Creates a data frame for each slice
    """
    Creates data frames for each slice type (eMBB, Medium, UrLLC) by reading in csv files from the respective directories.
    Returns a list of data frames and their corresponding lengths.
    """
    encoding = 'utf-8'
    # List CSV files from the specified directories for each slice type
    embbFiles = os.listdir('Slicing_UE_Data/Embb/')
    mediumFiles = os.listdir('Slicing_UE_Data/Medium/')
    urllcFiles = os.listdir('Slicing_UE_Data/Urllc/')

     # Read each CSV file into a list of data frames for each slice type
    dfEmbb = [pd.read_csv(f'Slicing_UE_Data/Embb/{file}', encoding=encoding) for file in embbFiles]
    dfMedium = [pd.read_csv(f'Slicing_UE_Data/Medium/{file}', encoding=encoding) for file in mediumFiles]
    dfUrllc = [pd.read_csv(f'Slicing_UE_Data/Urllc/{file}', encoding=encoding) for file in urllcFiles]

    # Define global variables to store the size of data frames
    global EMBB_DF_SIZE
    global MEDIUM_DF_SIZE
    global URLLC_DF_SIZE

    # Calculate the number of data frames for each slice type
    EMBB_DF_SIZE = len(dfEmbb)
    MEDIUM_DF_SIZE = len(dfMedium)
    URLLC_DF_SIZE = len(dfUrllc)

    # Calculate the length of each data frame in the lists
    embbFileLen = [len(fileDf) for fileDf in dfEmbb]
    mediumFileLen = [len(fileDf) for fileDf in dfMedium]
    urllcFileLen = [len(fileDf) for fileDf in dfUrllc]

    # Compact data frames and their lengths into lists
    df = [dfEmbb, dfMedium, dfUrllc]
    dfFileLen = [embbFileLen, mediumFileLen, urllcFileLen]

    # Return the data frames and their lengths
    return (df, dfFileLen)




