#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.

# Download pre-compiled files
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0rc3/mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz

# Unzip files
tar -xzvf mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz

# Go to the sdk folder
cd mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1/sdk

# Init environment
source env.sh

# If opencv 3+ is not installed on your system, execute the following command.
# If it is installed, skip this command
bash opencv.sh

# Compile executable programs
bash build.sh

# Go to mmdeploy folder
cd ../

# Download models
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-cpu.zip

# Unzip files
unzip rtmpose-cpu.zip

# Success
echo "Installation completed."
