#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.

INPUT_IMAGE=$1

./bin/pose_tracker \
    ./rtmpose-ort/rtmdet-nano \
    ./rtmpose-ort/rtmpose-m \
    $INPUT_IMAGE \
    --background black \
    --skeleton ./rtmpose-ort/t2i-adapter_skeleton.txt \
    --output ./skeleton_res.jpg \
    --pose_kpt_thr 0.4 \
    --show -1
