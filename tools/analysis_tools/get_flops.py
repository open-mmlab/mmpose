# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch

try:
    from fvcore.nn import (ActivationCountAnalysis, FlopCountAnalysis,
                           flop_count_str, flop_count_table, parameter_count)
except ImportError:
    print('You may need to install fvcore for flops computation, '
          'and you can use `pip install fvcore` to set up the environment')
from fvcore.nn.print_model_statistics import _format_size
from mmengine import Config

from mmpose.models import build_pose_estimator
from mmpose.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 192],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_pose_estimator(cfg.model)
    model.eval()

    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    inputs = (torch.randn((1, *input_shape)), )
    flops_ = FlopCountAnalysis(model, inputs)
    activations_ = ActivationCountAnalysis(model, inputs)

    flops = _format_size(flops_.total())
    activations = _format_size(activations_.total())
    params = _format_size(parameter_count(model)[''])

    flop_table = flop_count_table(
        flops=flops_,
        activations=activations_,
        show_param_shapes=True,
    )
    flop_str = flop_count_str(flops=flops_, activations=activations_)

    print('\n' + flop_str)
    print('\n' + flop_table)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n'
          f'Activation: {activations}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    register_all_modules()
    main()
