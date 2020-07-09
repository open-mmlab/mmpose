#!/usr/bin/env bash

node=pat_earth
numGPU=8
allGPU=8

CONFIG=configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py
WORK_DIR='work_dirs/hrnet_w32_coco_256x192/'

PY_ARGS=${@:5}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

OMPI_MCA_mpi_warn_on_fork=0 \
srun -p $node \
    --gres=gpu:$numGPU \
    -n$allGPU \
    --ntasks-per-node=$numGPU \
    --job-name=$expID \
    --kill-on-bad-exit=1 \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
