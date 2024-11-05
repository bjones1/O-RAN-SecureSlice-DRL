# DRL-Secure-Slicing

## Description
**DRL-Secure-Slicing** is a fully deployable O-RAN implementation featuring a network slicing xApp (extended application) operating within the Near-RT RIC (Real-Time Intelligent Controller). This project allows for the training of an intelligent agent in both emulated and real-world environments, utilizing automated scripts to streamline the process.

### Agent's Goal
The primary objective of the agent is to maximize throughput for each network slice while maintaining compliance with Service Level Agreements (SLAs). This involves ensuring that User Equipment (UE) operating within its SLA is securely sliced, effectively isolating it from other slices as necessary.

## System Model
![System Model](documentation/images/drl-ss-xapp-1.png)

### Development Environment
This project utilizes the CodeChat editor to facilitate a literate programming environment. Installation instructions can be found in the [CodeChat GitHub repository](https://github.com/bjones1/CodeChat_Editor).



## Getting Started

1. **Deploy srsRAN**  
   Follow the provided instructions [Instructions](documentation/DRL-SS-instructions.rst) to deploy srsRAN, choosing either the ZMQ (virtualized channel) option or USRPs connected to commercial off-the-shelf (COTS) equipment.

2. **Deploy the xApp**  
   Deploy the xApp using the same set of instructions.

3. **Run the DRL Agent**  
   Execute the DRL agent using either `emu` or `agent.py` to train the model within the specified environment.

4. **Model Checkpoints**  
   Model checkpoints are saved automatically, allowing you to integrate the trained model into the loop for real-time performance evaluation.
