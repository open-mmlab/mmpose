# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time
from datetime import datetime

import mmcv
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmpose.models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(
        description='running benchmark regression with tmux')

    parser.add_argument(
        '--config',
        '-c',
        help='test config file path',
        default='./.dev_scripts/benchmark/benchmark_cfg.yaml')

    parser.add_argument(
        '--priority',
        type=int,
        help='models with priority higher or equal to this will be included',
        default=2)

    # runtime setting parameters
    parser.add_argument(
        '--root-work-dir',
        '-r',
        help='the root working directory to store logs')

    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')

    parser.add_argument('--num-iters', default=50, help='test iterations')
    parser.add_argument('--num-warmup', default=5, help='warmup iterations')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.root_work_dir is None:
        # get the current time stamp
        now = datetime.now()
        ts = now.strftime('%Y_%m_%d_%H_%M')
        args.root_work_dir = f'work_dirs/speed_test_{ts}'
    mmcv.mkdir_or_exist(osp.abspath(args.root_work_dir))

    cfg = mmcv.load(args.config)

    results = []
    for i in range(args.priority + 1):
        models = cfg['model_list'][f'P{i}']
        for cur_model in models:
            cfg_file = cur_model['config']
            model_cfg = Config.fromfile(cfg_file)
            if 'input_shape' in cur_model.keys():
                input_shape = cur_model['input_shape']
                input_shape = tuple(map(int, input_shape.split(',')))
            else:
                image_size = model_cfg.data_cfg.image_size
                if isinstance(image_size, list):
                    input_shape = (3, ) + tuple(image_size)
                else:
                    input_shape = (3, image_size, image_size)

            model = build_posenet(model_cfg.model)
            model = model.cuda()
            model.eval()

            fp16_cfg = cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)
            if args.fuse_conv_bn:
                model = fuse_conv_bn(model)

            if hasattr(model, 'forward_dummy'):
                model.forward = model.forward_dummy
            else:
                raise NotImplementedError(
                    'FLOPs counter is currently not currently supported '
                    'with {}'.format(model.__class__.__name__))

            its, num_iters, pure_inf_time = get_model_inference_speed(
                model, input_shape, args.num_iters, args.num_warmup)

            split_line = '=' * 30
            result = f'{split_line}\nModel config:{cfg_file}\n' \
                     f'Input shape: {input_shape}\n' \
                     f'Overall average: {its:.2f} items / s\n' \
                     f'Total iters: {num_iters}\n'\
                     f'Total time: {pure_inf_time:.2f} s \n{split_line}\n'\

            print(result)
            results.append(result)

    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are included and verify that the '
          'speed computation is correct.')
    with open(osp.join(args.root_work_dir, 'speed.txt'), 'w') as f:
        for res in results:
            f.write(res)


def get_model_inference_speed(model, input_shape, num_iters=100, num_warmup=5):
    pure_inf_time = 0

    # benchmark with total batch and take the average
    for i in range(num_iters + num_warmup):

        try:
            batch = torch.ones(()).new_empty(
                (1, *input_shape),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device)
        except StopIteration:
            # Avoid StopIteration for models which have no parameters,
            # like `nn.Relu()`, `nn.AvgPool2d`, etc.
            batch = torch.ones(()).new_empty((1, *input_shape))

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            model(batch)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
    its = num_iters / pure_inf_time
    return its, num_iters, pure_inf_time


if __name__ == '__main__':
    main()
