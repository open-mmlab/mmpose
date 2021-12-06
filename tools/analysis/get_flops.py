# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from functools import partial

import torch

from mmpose.apis.inference import init_pose_model

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 192],
        help='input image size')
    parser.add_argument(
        '--input-constructor',
        '-c',
        type=str,
        choices=['none', 'batch'],
        default='none',
        help='If specified, it takes a callable method that generates '
        'input. Otherwise, it will generate a random tensor with '
        'input shape to calculate FLOPs.')
    parser.add_argument(
        '--batch-size', '-b', type=int, default=1, help='input batch size')
    parser.add_argument(
        '--not-print-per-layer-stat',
        '-n',
        action='store_true',
        help='Whether to print complexity information'
        'for each layer in a model')
    args = parser.parse_args()
    return args


def batch_constructor(flops_model, batch_size, input_shape):
    """Generate a batch of tensors to the model."""
    batch = {}

    img = torch.ones(()).new_empty(
        (batch_size, *input_shape),
        dtype=next(flops_model.parameters()).dtype,
        device=next(flops_model.parameters()).device)

    batch['img'] = img
    return batch


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = init_pose_model(args.config)

    if args.input_constructor == 'batch':
        input_constructor = partial(batch_constructor, model, args.batch_size)
    else:
        input_constructor = None

    if args.input_constructor == 'batch':
        input_constructor = partial(batch_constructor, model, args.batch_size)
    else:
        input_constructor = None

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(
        model,
        input_shape,
        input_constructor=input_constructor,
        print_per_layer_stat=(not args.not_print_per_layer_stat))
    split_line = '=' * 30
    input_shape = (args.batch_size, ) + input_shape
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
