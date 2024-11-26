# Table of Contents

## Contents

---

1. **[Readme](README.md)**: Overview and setup info.
2. **[Table of contents](toc.md)**: Repository overview.
3. **[Deployment instructions](documentation/DRL-SS-instructions.rst)**: Guide for deploying DRL-SS before training

---

<!-- ## NexRAN Source Files

1. **[Sources](src/)**: Core code files.
2. **[include](include/)**: Header files.
3. **[lib](lib/)**: Libraries and modules.
4. **[etc](etc/)**: Configuration files.

---

## xApp Specific Files

1. **[Dockerfile](Dockerfile)**: Docker image setup.
2. **[Onboard file](drl-ss-onboard.url)**: Onboarding metadata.
3. **[Config file](config-file.json)**: xApp settings.
4. **[Slicing configurations](zmqoneue.sh)**: Single UE slicing script.

--- -->

## DRL Code

1. **[Overview of DRL models](DRL-SSxApp/Overview.md)**: Description of objective, models, and how to inference
2. **[DQN agent training](DRL-SSxApp/DQN_agentemu.py)**: Agent training with DQN model
3. **[DDQN agent training](DRL-SSxApp/DDQN_agentemu.py)**: Agent training with DDQN model
4. **[Dueling DQN agent training](DRL-SSxApp/Dueling_DQN_agentemu.py)**: Agent training with Dueling DQN model
5. **[Inference Script for models](DRL-SSxApp/model_inference.py)**: A script tailored for inferencing model checkpoints

<!-- 1. **[Emulated environment](DRL-SSxApp/agentemu.py)**: Agent training without hardware.
2. **[Real environment](DRL-SSxApp/agent.py)**: Training with SDRs.
3. **[Setup script](DRL-SSxApp/setup_env.sh)**: Real environment setup.
4. **[Process termination](DRL-SSxApp/kill_stuff.sh)**: Ends active processes.
5. **[xApp interface](DRL-SSxApp/xapp_interface.py)**: xApp-environment interaction. -->

---

<!-- ### Additional Files

1. **[ZMQ broker](multi_ue.py)**: Multi-UE communication script. -->
