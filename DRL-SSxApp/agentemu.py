#!/usr/bin/env python3

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


"""
| Parameter             | Default Value | Description                                                                 | Try 1  |
|-----------------------|---------------|-----------------------------------------------------------------------------|--------| 
| `LR`                  | `3e-4`        | Learning Rate: Controls the step size during model weight updates.          | 
| `BATCH_SIZE`          | `128`         | num of experiences to sample from the replay buffer for each training batch.|
| `BUFFER_SIZE`         | `1e5`         | Size of the replay buffer that stores past experiences for learning.        |
| `UPDATE_EVERY`        | `8`           | Frequency of model parameter updates during training.                       |
| `TAU`                 | `1e-3`        | Soft update parameter for target network updates.                           |
| `EPISODE_MAX_TIMESTEP`| `300`         | Maximum number of timesteps per episode.                                    |
| `eps_start`           | `1.0`         | Initial value of epsilon for exploration vs. exploitation in action selectio|
| `eps_end`             | `0.01`        | Final value of epsilon after decay.                                         |
| `eps_decay`           | `0.995`       | Rate at which epsilon decays over time.                                     |
| `state_len`           | `8`           | Dimensionality of the state space.                                          |
| `action_len`          | `8`           | Dimensionality of the action space.                                         |
| `n_episodes`          | `1500`        | Number of episodes for training the agent.                                  |
| `max_t`               | `300`         | Maximum number of timesteps per episode.                                    |
"""

# Hyperparameters and environment-specific constants
LR = 5  # Learning Rate: Adjusts the model weights during training. Smaller values lead to slower training but more stable learning.
BATCH_SIZE = 128  # Number of experiences processed in each training batch.
BUFFER_SIZE = int(1e5)  # Size of the replay buffer, which stores past experiences for learning. Larger buffers can improve learning but require more memory.
UPDATE_EVERY = 8  # How often to update the model parameters during training.
TAU = 3e-4  # Soft update parameter for updating the target network. Determines how much of the local model's weights are copied to the target network.

# Constants for PRB and DL Bytes mapping and thresholds
PRB_INC_RATE = 6877         # Determines how many Physical Resource Blocks (PRB) to increase when adjusting PRB allocation.
# We expect 27223404
DL_BYTES_THRESHOLD = [19919004, 6640395, 664005] # Thresholds for different slice types: Embb, Medium, Urllc.
#DL_BYTES_THRESHOLD = [6640395, 6640395, 6640395] # Thresholds for different slice types: Embb, Medium, Urllc.

REWARD_PER_EPISODE = []  # List to store rewards for each episode
EPISODE_MAX_TIMESTEP = 300  # Maximum number of timesteps per episode

class QNetwork(nn.Module):
    """
    Q-Network for approximating the Q-value function.
    The Q-network predicts Q-values for each action given a state.
    """
    def __init__(self, state_len, action_len, seed, layer1_size=128, layer2_size=128, layer3_size=128, layer4_size=128):
        super(QNetwork, self).__init__()
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

class Agent():
    """
    Agent that interacts with the environment and learns from it using a Q-learning approach.
    """
    def __init__(self, state_len, action_len, seed, DDQN=False):
        self.action_len = action_len
        self.state_len = state_len
        self.seed = random.seed(seed)

        # Initialize the local and target Q-networks
        self.qnetwork_local = QNetwork(state_len, action_len, seed).to(device)
        self.qnetwork_target = QNetwork(state_len, action_len, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_len, BUFFER_SIZE, BATCH_SIZE)
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

class ReplayBuffer:
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

def dqn(n_episodes=1500, max_t=300, eps_start=1.0, eps_end=0.01, eps_decay=0.995, pth_file='checkpoint.pth'):
    """
    Train the DQN agent with specified parameters and save checkpoints.
    """
    eps = eps_start  # Initialize epsilon (exploration rate)
    df, df_file_len = create_df()  # Create the dataframes for each slice and get their lengths

    actions = [] 
    rewards = []
    total_prbs = [[],[],[]]
    dl_bytes = [[],[],[]]
    action_prbs = [1,1,1] # agent defined number of prbs for each state
    total = 0

    total = 0
    correct = 0
    reward_averages = list()
    percentages = list()
    action_count = [0 for x in range(7)]
    EPISODE_MAX_TIMESTEP = max_t

    for episode in range(1, n_episodes + 1):
        done = False
        temp = [-1] * EPISODE_MAX_TIMESTEP
        max_t = 0
        i = 1
        # assigned PRBs to each slice
        action_prbs = [1,1,1]
        global DL_BYTE_TO_PRB_RATES
        # number of DL bytes that each slice increases by per PRB
        DL_BYTE_TO_PRB_RATES = [10000, 10000, 10000]

    
        next_state = get_state(df, df_file_len, action_prbs)  # Get the initial state from the dataframes
        while not done and max_t < EPISODE_MAX_TIMESTEP:
            total += 1
            state = next_state
            action = agent.act(np.array(state), eps)  # Choose an action based on the current state
            action_count[action] += 1

            for i, prb_value in enumerate(total_prbs):
                total_prbs[i].append(prb_value)
                dl_bytes[i].append(state[i])

            reward, done, action_prbs = perform_action(action, state, i, action_prbs)
            actions.append(action)
            rewards.append(reward)
            if reward > 0:
                correct += 1
            next_state = get_state(df, df_file_len, action_prbs)  # Get the initial state from the dataframes
            agent.step(state, action, reward, next_state, done)  # Update the agent with the experience

            # Update the score with the reward
            max_t += 1  # Increment the timestep
            i += 1  # Increment the step counter

        if episode % 100 == 0:  # Decay epsilon every 500 episodes
            eps = max(eps_end, eps_decay * eps)
            
        # Store the score and actions
        actions.append(temp)

        # Update the state lists with the current state

        # Logging progress
        avg_reward = sum(rewards) / len(rewards)
        print(f'\rEpisode {episode}\tAverage Score: {avg_reward:.2f}', end="")
        reward_averages.append(avg_reward)
        percentages.append(correct/total)
        if episode % 50000 == 0:
            print(f'\rEpisode {episode}\tAverage Score: {avg_reward}')
            print("Percentage: ", correct / total)
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))

            ax[0].plot(reward_averages, 'r')
            ax[0].set_title('Reward')

            ax[1].plot(percentages, color='g')
            ax[1].set_title('Percentages')

            ax[2].bar(range(len(action_count)), action_count, color='b')
            ax[2].set_title('Actions taken')

            action_count = [0 for x in range(7)]

            plt.show()


    # Save the model checkpoint
    torch.save(agent.qnetwork_local.state_dict(), pth_file)
    print(f'Model saved to {pth_file}')

    return rewards, (correct, total)
                

            


def create_df():  # Creates a data frame for each slice
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

def get_state(df, df_file_len, action_prbs):
    # Randomly select one file from each slice type
    randEmbbFile = np.random.randint(0, EMBB_DF_SIZE)
    randMediumFile = np.random.randint(0, MEDIUM_DF_SIZE)
    randUrllcFile = np.random.randint(0, URLLC_DF_SIZE)

    # Randomly select one row from the chosen file for each slice type
    randEmbbData = df[0][randEmbbFile].iloc[np.random.randint(0, df_file_len[0][randEmbbFile])]
    randMediumData = df[1][randMediumFile].iloc[np.random.randint(0, df_file_len[1][randMediumFile])]
    randUrllcData = df[2][randUrllcFile].iloc[np.random.randint(0, df_file_len[2][randUrllcFile])]

    # Print column names for debugging
    # print("Embb columns:", df[0][randEmbbFile].columns)
    # print("Medium columns:", df[1][randMediumFile].columns)
    # print("Urllc columns:", df[2][randUrllcFile].columns)

    # Ensure columns exist and handle missing columns gracefully
    embb_bytes = float(randEmbbData.get('dl_bytes', 0))
    #embb_prbs = float(randEmbbData.get('dl_prbs', 1))
    
    medium_bytes = float(randMediumData.get('dl_bytes', 0))
    #medium_prbs = float(randMediumData.get('dl_prbs', 1))
    
    urllc_bytes = float(randUrllcData.get('dl_bytes', 0))
    #urllc_prbs = float(randUrllcData.get('dl_prbs', 1))

    # Every time step there is a chance one slice becomes malicous (small)
    # if a slice is malicous the DL bytes will go way above the threashold
    chance = random.randint(0,300000)


    global DL_BYTE_TO_PRB_RATES
    if chance == 1000:
        DL_BYTE_TO_PRB_RATES[0] *= 10
    elif chance == 2000:
        DL_BYTE_TO_PRB_RATES[1] *= 10
    elif chance == 3000:
        DL_BYTE_TO_PRB_RATES[2] *= 10




    return [DL_BYTE_TO_PRB_RATES[0] * action_prbs[0],
            DL_BYTE_TO_PRB_RATES[1] * action_prbs[1],
            DL_BYTE_TO_PRB_RATES[2] * action_prbs[2]]


def perform_action(action, state, i, action_prbs):
    """
    If action is 0: Increase PRB in slice 1

    If action is 1: Increase PRB in slice 2


    If action is 2: Increase PRB in slice 3


    If action is 3: Its optimal



    """
    reward = 0
    done = False
    next_state = np.copy(state)
    if sum(action_prbs) == 0:
        done = True
        return reward, done, action_prbs
    if action <= 3:
        for i, dl_bytes_value in enumerate(state):
            if dl_bytes_value > DL_BYTES_THRESHOLD[i]:
                action_prbs[i] -= 5
                reward += 0
            else:
                reward += dl_bytes_value
        reward /= len(state)
        
        if action <=2:
            if action_prbs[action] == 0:
                reward += 0
            else:
                action_prbs[action] += 5

    else:
        action_prbs[action - 4] = 0
        if state[action - 4] > DL_BYTES_THRESHOLD[action - 4]:
            reward += max(state)
        else:
            reward += 0
    


    return reward, done, action_prbs

# Define the state size and action size for the agen10100,t
state_size = 3
action_size = 7  # Actions: Increase PRB, Decrease PRB, Secure Slice

# Initialize the agent
agent = Agent(state_size, action_size, seed=0, DDQN=False)

# With 1000 max_t mathematically every slice should become malicious in every episode at some point
rewards, percent = dqn(n_episodes=1000000, max_t=3000, eps_start=1.0, eps_end=0.01, eps_decay=0.999, pth_file='checkpoint.pth')

# Print test results
print("Tests correct: " + str(percent[0]))
print("Tests incorrect: " + str(percent[1]))
