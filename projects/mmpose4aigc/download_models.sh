#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.

# Create models folder
mkdir models

# Go to models folder
cd models

# Download det model
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth

# Download pose model
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth

# Go back mmpose4aigc
cd ..

# Success
echo "Download completed."
