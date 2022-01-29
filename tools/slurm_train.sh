#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

set -x

version_p=$(python -c 'import sys; print(sys.version_info[:])')
CONDA_ROOT=/mnt/cache/share/platform/env/miniconda3.${version_p:4:1}
MMCV_PATH=${CONDA_ROOT}/envs/${CONDA_DEFAULT_ENV}/mmcvs
mmcv_version=1.4.2
export PYTHONPATH=${MMCV_PATH}/${mmcv_version}:$PYTHONPATH
export MMCV_HOME=/mnt/lustre/share_data/PAT/datasets/pretrain/mmcv

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
