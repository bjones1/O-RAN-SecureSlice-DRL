# Table of Contents

## Contents

1. **[Readme](README.md)**: This file provides an overview of the project, including key information about its purpose, structure, and any initial setup or usage instructions. It serves as an entry point for users to understand what the repository contains.

2. **[Table of contents](toc.md)**: This file lists and organizes the contents of the repository. It helps users navigate the various files and sections within the project by providing a structured overview.

3. **[Deployment instructions](documentation/DRL-SS-instructions.rst)**: A detailed guide on how to deploy the DRL-SS (Deep Reinforcement Learning-based Slice Scheduler) system. It covers the setup steps, configuration, and troubleshooting tips necessary to deploy the solution.



---

## NexRAN Source Files

1. **[Sources](src/)**: Contains the core source code for NexRAN, including scripts, modules, and other code files essential for the functionality of the system.

2. **[include](include/)**: This directory holds the header files or definitions used in the NexRAN source code. These files provide essential declarations and structures shared across multiple components.

3. **[lib](lib/)**: The library directory includes compiled libraries or additional modules required by NexRAN to function, offering supplementary functionality and dependencies.

4. **[etc](etc/)**: This folder contains configuration files or other settings that define the behavior of the NexRAN system, allowing customization and fine-tuning of parameters.

---

## xApp Specific Files

1. **[Dockerfile](Dockerfile)**: This file defines the Docker image for the xApp, specifying the environment, dependencies, and setup instructions to build a containerized version of the application.

2. **[Onboard file](drl-ss-onboard.url)**: A URL or metadata file used to onboard the xApp into a specific environment or platform, indicating its source and version.

3. **[Config file](config-file.json)**: The main configuration file for the xApp, providing key settings and parameters that influence its operation and integration with other components.

4. **[Slicing configurations](zmqoneue.sh)**: A shell script that configures slicing parameters for a single UE setup. This script likely manages the network slice configuration through ZMQ, optimizing connectivity for a single UE scenario.

---

## DRL code


1. **[Emulated environment for training agent](DRL-SSxApp/agentemu.py)**: This script provides an emulated environment for training the agent, allowing experimentation and testing without requiring physical hardware.

2. **[Real environment for training agent (Requires SDRs)](DRL-SSxApp/agent.py)**: This script sets up a real environment for training the agent, requiring Software-Defined Radios (SDRs) for actual deployment and interaction.

3. **[Setup script for real environment](DRL-SSxApp/setup_env.sh)**: A setup script that configures the necessary environment for the real agent training setup, including dependencies and initial configurations.

4. **[Script to end processes](DRL-SSxApp/kill_stuff.sh)**: This script terminates active processes related to the agent's environment, ensuring a clean shutdown of components and resources.

5. **[xApp interactions with real environment](DRL-SSxApp/xapp_interface.py)**: This script handles interactions between the xApp and the real environment, facilitating data exchange and control commands in a live scenario.


### Additional Files

1. **[ZMQ broker for multi UE](multi_ue.py)**: This Python script sets up a ZeroMQ (ZMQ) broker for handling multiple user equipment (UE) connections. It enables communication between various network components in a multi-UE scenario.