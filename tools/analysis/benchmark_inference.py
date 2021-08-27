#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPose benchmark a recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--log-interval', default=10, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model = MMDataParallel(model, device_ids=[0])

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with total batch and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            model(return_loss=False, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                its = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done item [{i + 1:<3}],  {its:.2f} items / s')
    print(f'Overall average: {its:.2f} items / s')
    print(f'Total time: {pure_inf_time:.2f} s')


if __name__ == '__main__':
    main()
