#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
"""This file is for benchmark data loading process. It can also be used to
refresh the memcached cache. The command line to run this file is:

$ python -m cProfile -o program.prof tools/analysis/benchmark_processing.py
configs/task/method/[config filename]

Note: When debugging, the `workers_per_gpu` in the config should be set to 0
during benchmark.

It use cProfile to record cpu running time and output to program.prof
To visualize cProfile output program.prof, use Snakeviz and run:
$ snakeviz program.prof
"""
import argparse

import mmcv
from mmcv import Config

from mmpose import __version__
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import get_root_logger


def main():
    parser = argparse.ArgumentParser(description='Benchmark data loading')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # init logger before other steps
    logger = get_root_logger()
    logger.info(f'MMPose Version: {__version__}')
    logger.info(f'Config: {cfg.text}')

    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # Start progress bar after first 5 batches
    prog_bar = mmcv.ProgressBar(
        len(dataset) - 5 * cfg.data.samples_per_gpu, start=False)
    for i, data in enumerate(data_loader):
        if i == 5:
            prog_bar.start()
        for _ in data['img']:
            if i < 5:
                continue
            prog_bar.update()


if __name__ == '__main__':
    main()
