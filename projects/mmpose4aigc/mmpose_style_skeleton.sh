#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.

WORKSPACE=mmdeploy-1.0.0-linux-x86_64-cxx11abi
export LD_LIBRARY_PATH=${WORKSPACE}/lib:${WORKSPACE}/thirdparty/onnxruntime/lib:$LD_LIBRARY_PATH

INPUT_IMAGE=$1

${WORKSPACE}/bin/pose_tracker \
    ${WORKSPACE}/rtmpose-ort/rtmdet-nano \
    ${WORKSPACE}/rtmpose-ort/rtmpose-m \
    $INPUT_IMAGE \
    --background black \
    --skeleton ${WORKSPACE}/rtmpose-ort/t2i-adapter_skeleton.txt \
    --output ./skeleton_res.jpg \
    --pose_kpt_thr 0.4 \
    --show -1
