#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.

# Download pre-compiled files
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-linux-x86_64-cxx11abi.tar.gz

# Unzip files
tar -xzvf mmdeploy-1.0.0-linux-x86_64-cxx11abi.tar.gz

# Go to the sdk folder
cd mmdeploy-1.0.0-linux-x86_64-cxx11abi

# Init environment
source set_env.sh

# If opencv 3+ is not installed on your system, execute the following command.
# If it is installed, skip this command
bash install_opencv.sh

# Compile executable programs
bash build_sdk.sh

# Download models
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-cpu.zip

# Unzip files
unzip rtmpose-cpu.zip

# Success
echo "Installation completed."
