#!/usr/bin/env python3

import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from xapp_interface import KpmInterface, IperfInterface, ConfInterface
import time
from collections import namedtuple, deque
import subprocess
import os
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
DULL_GREY = '\033[90m'
ENDC = '\033[0m'

LR = 5e-4
BATCH_SIZE = 20
BUFFER_SIZE = int(1e5)
UPDATE_EVERY = 4
TAU = 1e-3

start_flag = False
env_pid = 0

# Signal handlers for running the environment
def finish_handler(sig, frame):
    global start_flag
    if sig == signal.SIGUSR1:
        print(YELLOW + "\n\nStart Signal Recieved!\n\n" + ENDC)
        start_flag = True

# Exit handler to kill stray processes
def exit_handler(sig, frame):
    print(RED + "SIGINT Recieved, exiting..." + ENDC)
    subprocess.Popen(['sudo', 'bash', './kill_stuff.sh'])
    sys.exit(0)


# Runs the script and waits starts iperf
def setup_env(iperf_i):
    print(GREEN + "Starting xApp..." + ENDC)
    env_process = subprocess.Popen(['sudo', 'bash', './setup_env.sh'], stdout=subprocess.PIPE)
    global env_pid
    env_pid = env_process.pid
    wait_for_finish(env_process.stdout, iperf_i)

# Wait for SIGUSR1
def wait_for_finish(process_pipe, iperf_i):
    global start_flag
    while(not start_flag):
        line = process_pipe.readline()
        if line.decode().strip() == "iperf3 -s -B 172.16.0.1 -p 5030 -i 1":
            print(YELLOW + "Starting Iperf Client" + ENDC)
            iperf_i.start()
        print(DULL_GREY + "\t" + line.decode().strip() + ENDC)
    print(GREEN + "Script Finished" + ENDC)



# Kills ENV process to restart 
def kill_env():
    global env_pid
    os.kill(env_pid, signal.SIGINT)


signal.signal(signal.SIGINT, exit_handler)
signal.signal(signal.SIGUSR1, finish_handler)

class KpiEvm:

    def __init__(self):
        """
        Interface Setup:
        KpmInterface => gets all necessary KPM readings
        IperfInterface => gets the thorughput with a background iperf process
        ConfInterface => allows for binding of ues and prb allocation
        """
        self.namespaces = ["ue1", "ue2", "ue3"]
        print('Starting KPMInterface.....')
        self.kpm_i = KpmInterface()
        print('Starting ConfInterface......')
        self.conf_i = ConfInterface()
        print('Starting IperfInterface.....')
        self.iperf_i = IperfInterface(self.namespaces)

        setup_env(self.iperf_i)
        
        self.slice = "fast"
        self.ues = list()
        self.ues += [ue for ue in self.conf_i.get_slice(self.slice)["ues"]]
        print("Ues Detected: ", self.ues)
        self.sla = 10.00
        print("SLA: ", self.sla)
        self.prbs = self.conf_i.get_slice(self.slice)["allocation_policy"]["share"]
        self.previous_tp = [0 for _ in range(len(self.namespaces))]
        self.reset()

    def get_current_reward(self):
        print("Reward: ", end="")
        reward = 0
        current_tp = list()
        for i in range(len(self.namespaces)):
            current_tp.append(self.iperf_i.get_reading(self.namespaces[i]))
            if self.previous_tp[i] > current_tp[i]:
                reward -= 10
            elif self.previous_tp[i] <= current_tp[i]:
                reward += 10

        for namespace in self.namespaces:
            if self.iperf_i.get_reading(namespace)  > self.sla:
                reward -= 100

        self.previous_tp = current_tp
        print(YELLOW + str(reward) + ENDC)
        return reward

    def get_current_state(self):
        print("State: ", end="")
        """
        State is defined as:
        1. the most recent kpms of each ue
        2. the most recent throughput reading
        3. the current total prbs in the slice
        """
        state = list()
        recieved_data = self.kpm_i.get_kpms()
        ue_ids = list()
        for data_packet in recieved_data:
            for ue_name, kpm in sorted(data_packet.items()):
                if ue_name in ue_ids:
                    continue
                ue_ids.append(ue_name)
                state += [np.int64(int(value)) for value in kpm.values()]
        state.append(self.prbs)
        state += [self.iperf_i.get_reading(namespace) for namespace in self.namespaces]
        print(GREEN + str(state) + ENDC)
        return state

    def reset(self):
        return self.get_current_state() , self.get_current_reward(), False

    def restart(self):
        # This function should be run to reset the interfaces after the env is restarted
        print(RED + "Restarting KpiEvm" + ENDC)
        self.prbs = self.conf_i.get_slice(self.slice)["allocation_policy"]["share"]
        self.previous_tp = [0 for _ in range(len(self.namespaces))]

    def step(self, act):
        """
        Perform an action:
        Action 1 => increase PRB allocation to slice
        Action 2 => decrease PRB allocation to slice
        Action 3 => move malicious UE to secure slice
        """
        print(BLUE + f"Taking Action {act}" + ENDC)
        # Note that action of 0 means no action will be taken
        if act == 1:
            self.prbs += 10
            self.conf_i.reallocate_prbs(self.prbs, self.slice)

        elif act == 2:
            self.prbs -= 10
            self.conf_i.reallocate_prbs(self.prbs, self.slice)

        elif act == 3:
            largest = 0
            for i in range(len(self.namespaces)):
                if self.iperf_i.get_reading(self.namespaces[i]) > self.iperf_i.get_reading(self.namespaces[largest]):
                    largest = i
            # XXX: a temporary workaround
            namespace_to_imsi = {
                "ue1":"001010123456789",
                "ue2":"001010123456780",
                "ue3":"001010123456781"
            }
           #self.conf_i.unbind_ue(namespace_to_imsi[self.namespaces[largest]], self.slice)
           #self.conf_i.bind_ue_to_slice(namespace_to_imsi[self.namespaces[largest], "secure_slice"])

        return self.get_current_state(), self.get_current_reward(), False



class QNetwork(nn.Module):
    
    def __init__(self, state_len, action_len, seed, layer1_size=64, layer2_size=64):
        """
        Parameters:
        state_len (int) => size of the input state
        action_len (int) => size of the action space
        seed (int) => random seed
        layer1_size (int) => number of neurons in layer 1
        layer2_size (int) => number of neurons in layer 2
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.l1 = nn.Linear(state_len, layer1_size)
        self.l2 = nn.Linear(layer1_size, layer2_size)
        self.l3 = nn.Linear(layer2_size, action_len)

    def forward(self, input_state):
        # Param: input state (int) => state to train
        x = F.relu(self.l1(input_state))
        x = F.relu(self.l2(x))
        return self.l3(x)

class Agent():

    def __init__(self, state_len, action_len, seed, DDQN=False):
        """
        Parameters:
        state_len (int) => size of the input state
        action_len (int) => size of the action space
        seed (int) => random seed
        DDQN (bool) => double DQN?
        """
        self.action_len = action_len
        self.state_len = state_len
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_len, action_len, seed).to(device)
        self.qnetwork_target = QNetwork(state_len, action_len, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_len, BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0
        self.DDQN = DDQN

    def step(self, state, action, reward, next_state, done):
        # add experience to memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, 0.99) # learn based on discount factor

    def act(self, state, eps=0.):
        """
        Params:
        state (array) => current state
        eps (float) => epsilon action selection
        """
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
        """
        Params:
        experiences (tuple) => tuple of s, a, r, s', done
        gamma (float) => discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        print(actions)
        print(states)
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        Q_targets_next = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    def soft_update(self, local_model, target_model ,tau):
        """
        Params:
        local_model (model) => copy weights from
        target_model (model) => copy weights to 
        tau (float) => interpolation
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0-tau) * target_param.data)


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size):
        """
        Params:
        action_size (int) => dimensions of actions
        buffer_size (int) => max buffer size
        batch_size (int) => size of training batch
        seed (int) => random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.random()

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

def dqn(n_episodes=1500, max_t=300, eps_start=1.0, eps_end=0.01, eps_decay=0.995,pth_file = 'checkpoint.pth'):
    eps=eps_start                                         # initialize the score
    scores_window = deque(maxlen=100)  # last 100 scores
    scores = []
    print('Starting of agent training ......')
    for episode in range(1,n_episodes+1):
        next_state, reward, done = env.reset() # reset the environment
        state = next_state           # get the current state
        score = 0
        for time_step in range(max_t):
            action = agent.act(np.array(state),eps)        # select an action
            next_state, reward, done = env.step(action)        # send the action to the environment
            agent.step(state, action, reward, next_state, done)

            score += reward                                # update the score

            state = next_state                             # roll over the state to next time step
            eps = max(eps_end,eps_decay*eps)
            if done:
                break
        scores.append(score)
        scores_window.append(score)       # save most recent score

        kill_env()
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        if np.mean(scores_window)>=5000:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), pth_file)
            # model = torch.load(PATH)
            break
        setup_env(env.iperf_i)
        env.restart()
    return scores,reward



state_size = 23
action_size = 4
agent = Agent(state_size, action_size, False)
env = KpiEvm()

start_time = time.time()
scores_dqn_base, reward = dqn(pth_file="test.pth")


# https://github.com/tkcoding/Stock_DRL/blob/main/Stock%20Price%20DQN%20.ipynb




