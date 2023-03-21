#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.

WORKSPACE=mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1/sdk/
export LD_LIBRARY_PATH=${WORKSPACE}/lib:${WORKSPACE}/thirdparty/onnxruntime/lib:$LD_LIBRARY_PATH

INPUT_IMAGE=$1

mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1/sdk/bin/pose_tracker \
    mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1//rtmpose-ort/rtmdet-nano \
    mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1//rtmpose-ort/rtmpose-m \
    $INPUT_IMAGE \
    --background black \
    --skeleton mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1//rtmpose-ort/t2i-adapter_skeleton.txt \
    --output ./skeleton_res.jpg \
    --pose_kpt_thr 0.4 \
    --show -1
