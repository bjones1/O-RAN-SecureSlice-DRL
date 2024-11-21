from DQN_agentemu import DQN, DQN_QNetwork, run_dqn
from DDQN_agentemu import DDQN, DDQN_QNetwork, run_ddqn
from Dueling_DQN_agentemu import DQN_Dueling, Dueling_QNetwork, run_dueling
import argparse
import sys
import pandas as pd
import torch
import numpy as np
from common import get_state
import random
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def parse():
    """
    Reads in CLI arguments
    Returns argparse arguments object
    """
    parser = argparse.ArgumentParser(
        description="Run an inference on DQN DDQN and Dueling DQN models")
    parser.add_argument(
        "--operation",
        type=str,
        default="inference",
        help="Operation to perform on the chosen model. Options: inference | train")
    parser.add_argument(
        "--model_type",
        type=str,
        default="DQN",
        help="Type of model to use. Options: DQN, DDQN, Dueling")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=60000,
        help="The number of episodes to run")
    parser.add_argument("--malicious_chance",
                        type=int,
                        default=100,
                        help="The chance of any UE to become malicious in one timestep (1/malicious_chance chance)")
    parser.add_argument("--malicious_chance_increase",
                        type=float,
                        default=0,
                        help="The rate of increase of the malicious chance (malicious_chance+=malicious_chance_increase)")
    return parser.parse_args()

def train(args):
    state_size = 3
    action_size = 4  # Actions: Increase PRB, Decrease PRB, Secure Slice

    if args.model_type == "DQN":
        agent = DQN(state_size, action_size, seed=0, DDQN=False)

        rewards, percent = run_dqn(
            agent,
            n_episodes=args.num_episodes,
            max_t=4,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay=0.99,
            pth_file='DQNcheckpoint.pth',
            malicious_chance=args.malicious_chance,
            malicious_chance_increase=args.malicious_chance_increase
        )

        print("Tests correct: " + str(percent[0]))
        print("Tests incorrect: " + str(percent[1]))

        rewards_df = pd.DataFrame(rewards, columns=["Reward"])
        rewards_df.to_csv("training_rewards/DQN_episode_rewards.csv", index=False)
        print("Rewards saved to DQN_episode_rewards.csv")

    elif args.model_type == "DDQN":
        agent = DDQN(state_size, action_size, seed=0, DDQN=True)

        rewards, percent = run_ddqn(
            agent,
            n_episodes=args.num_episodes,
            max_t=4,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay=0.99,
            pth_file='DDQNcheckpoint.pth',
            malicious_chance=args.malicious_chance,
            malicious_chance_increase=args.malicious_chance_increase
        )

        print("Tests correct: " + str(percent[0]))
        print("Tests incorrect: " + str(percent[1]))

        rewards_df = pd.DataFrame(rewards, columns=["Reward"])
        rewards_df.to_csv("training_rewards/DDQN_episode_rewards.csv", index=False)
        print("Rewards saved to DDQN_episode_rewards.csv")

    elif args.model_type == "Dueling":
        agent = DQN_Dueling(state_size, action_size, seed=0, DDQN=True)

        rewards, percent = run_dueling(
            agent,
            n_episodes=args.num_episodes,
            max_t=4,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay=0.99,
            pth_file='Dueling_DQNcheckpoint.pth',
            malicious_chance=args.malicious_chance,
            malicious_chance_increase=args.malicious_chance_increase
        )

        print("Tests correct: " + str(percent[0]))
        print("Tests incorrect: " + str(percent[1]))

        rewards_df = pd.DataFrame(rewards, columns=["Reward"])
        rewards_df.to_csv("training_rewards/Dueling_episode_rewards.csv", index=False)
        print("Rewards saved to Dueling_episode_rewards.csv")

    return 0

def get_action(agent, state):
    with torch.no_grad():
        action_values = agent(state)
    return np.argmax(action_values.cpu().data.numpy())

def run_inference_epoch(agent, num_episodes, malicious_chance):
    action_prbs = [2897, 965, 91]  # eMBB, Medium, URLLC
    global DL_BYTE_TO_PRB_RATES
    DL_BYTE_TO_PRB_RATES = [6877, 6877, 6877]
    is_mal = False
    incorrect_actions = 0

    for i in range(num_episodes):
        # Only trigger malicious behavior if malicious_chance > 0
        if malicious_chance > 0 and random.randint(0, int(malicious_chance)) == malicious_chance:
            is_mal = True
            # Introduce malicious event by multiplying by 10
            DL_BYTE_TO_PRB_RATES[random.randint(0, 2)] *= 10

        # If no malicious event, keep the state as it is
        state = [DL_BYTE_TO_PRB_RATES[0] * action_prbs[0],
                 DL_BYTE_TO_PRB_RATES[1] * action_prbs[1],
                 DL_BYTE_TO_PRB_RATES[2] * action_prbs[2]]

        # Debugging: print state before conversion
        print(f"State at episode {i}: {state}")

        # Ensure state is valid
        if any(val is None or not isinstance(val, (int, float)) for val in state):
            print(f"Invalid state detected: {state}")
            continue  # Skip this episode if the state is invalid

        # Use np.float64 to handle large numbers better
        np_state = np.array(state, dtype=np.float64)
        state = torch.from_numpy(np_state).float().unsqueeze(0).to(device)
        selected_action = get_action(agent, state)

        if selected_action >= 3 and not is_mal:
            incorrect_actions += 1
        elif selected_action < 3 and is_mal:
            incorrect_actions += 1
            is_mal = False

    return 1 - (incorrect_actions / num_episodes)




def inference(args):
    state_size = 3
    action_size = 4
    results = None

    if args.model_type == "DQN":
        agent = DQN_QNetwork(state_size, action_size, seed=0).to(device)
        state_dict = torch.load("pth/DQNcheckpoint.pth", map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
        results = run_inference_epoch(agent, args.num_episodes, args.malicious_chance)

    elif args.model_type == "DDQN":
        agent = DDQN_QNetwork(state_size, action_size, seed=0).to(device)
        state_dict = torch.load("pth/DDQNcheckpoint.pth", map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
        results = run_inference_epoch(agent, args.num_episodes, args.malicious_chance)

    elif args.model_type == "Dueling":
        agent = Dueling_QNetwork(state_size, action_size, seed=0).to(device)
        state_dict = torch.load("pth/Dueling_DQNcheckpoint.pth", map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
        results = run_inference_epoch(agent, args.num_episodes, args.malicious_chance)

    print(f"Inference result: {results}")

    # Write results to CSV file
    save_inference_results(args.model_type, args.malicious_chance, args.num_episodes, results)

    return 0

def save_inference_results(model_type, malicious_chance, num_episodes, results):
    # Ensure the directory exists
    os.makedirs('reward_data', exist_ok=True)
    
    # Prepare the data to save
    data = {
        'Model': [model_type],
        'Malicious Chance': [malicious_chance],
        'Num Episodes': [num_episodes],
        'Result': [results]
    }
    df = pd.DataFrame(data)

    # Define the file path
    file_path = 'reward_data/inference_results.csv'

    # Append to the CSV file if it exists, otherwise create a new one
    if os.path.isfile(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)

    print(f"Results saved to {file_path}")

def main():
    args = parse()

    if args.operation == "train":
        return train(args)
    elif args.operation == "inference":
        return inference(args)

if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
