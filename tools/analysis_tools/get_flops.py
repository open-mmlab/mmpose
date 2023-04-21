# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
from mmengine.config import DictAction
from mmengine.logging import MMLogger

from mmpose.apis.inference import init_model

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get complexity information from a model config')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--device', default='cpu', help='Device used for model initialization')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=[256, 192],
        help='input image size')
    parser.add_argument(
        '--batch-size',
        '-b',
        type=int,
        default=1,
        help='Input batch size. If specified and greater than 1, it takes a '
        'callable method that generates a batch input. Otherwise, it will '
        'generate a random tensor with input shape to calculate FLOPs.')
    parser.add_argument(
        '--show-arch-info',
        '-s',
        action='store_true',
        help='Whether to show model arch information')
    args = parser.parse_args()
    return args


def batch_constructor(flops_model, batch_size, input_shape):
    """Generate a batch of tensors to the model."""
    batch = {}

    inputs = torch.randn(batch_size, *input_shape).new_empty(
        (batch_size, *input_shape),
        dtype=next(flops_model.parameters()).dtype,
        device=next(flops_model.parameters()).device)

    batch['inputs'] = inputs
    return batch


def inference(args, input_shape, logger):
    model = init_model(
        args.config,
        checkpoint=None,
        device=args.device,
        cfg_options=args.cfg_options)

    if hasattr(model, '_forward'):
        model.forward = model._forward
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    if args.batch_size > 1:
        outputs = {}
        avg_flops = []
        logger.info('Running get_flops with batch size specified as {}'.format(
            args.batch_size))
        batch = batch_constructor(model, args.batch_size, input_shape)
        for i in range(args.batch_size):
            result = get_model_complexity_info(
                model,
                input_shape,
                inputs=batch['inputs'],
                show_table=True,
                show_arch=args.show_arch_info)
            avg_flops.append(result['flops'])
        mean_flops = _format_size(int(np.average(avg_flops)))
        outputs['flops_str'] = mean_flops
        outputs['params_str'] = result['params_str']
        outputs['out_table'] = result['out_table']
        outputs['out_arch'] = result['out_arch']
    else:
        outputs = get_model_complexity_info(
            model,
            input_shape,
            inputs=None,
            show_table=True,
            show_arch=args.show_arch_info)
    return outputs


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    if len(args.input_shape) == 1:
        input_shape = (3, args.input_shape[0], args.input_shape[0])
    elif len(args.input_shape) == 2:
        input_shape = (3, ) + tuple(args.input_shape)
    else:
        raise ValueError('invalid input shape')

    if args.device == 'cuda:0':
        assert torch.cuda.is_available(
        ), 'No valid cuda device detected, please double check...'

    outputs = inference(args, input_shape, logger)
    flops = outputs['flops_str']
    params = outputs['params_str']
    split_line = '=' * 30
    input_shape = (args.batch_size, ) + input_shape
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print(outputs['out_table'])
    if args.show_arch_info:
        print(outputs['out_arch'])
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
