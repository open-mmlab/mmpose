# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmpose.apis.inference import init_model

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device used for model initialization')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
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
        '--not-show-complexity-table',
        '-n',
        action='store_true',
        help='Whether to show complexity information')
    args = parser.parse_args()
    return args


def batch_constructor(flops_model, batch_size, input_shape):
    """Generate a batch of tensors to the model."""
    batch = {}

    inputs = torch.ones(()).new_empty(
        (batch_size, *input_shape),
        dtype=next(flops_model.parameters()).dtype,
        device=next(flops_model.parameters()).device)

    batch['inputs'] = inputs
    return batch


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    if args.device == 'cuda:0':
        assert torch.cuda.is_available(
        ), 'No valid cuda detected, please double check...'

    model = init_model(
        args.config,
        checkpoint=None,
        device=args.device,
        cfg_options=args.cfg_options)

    if args.input_constructor == 'batch':
        input_constructor = partial(batch_constructor, model, args.batch_size)
    else:
        input_constructor = None

    if hasattr(model, '_forward'):
        model.forward = model._forward
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmpose'))

    outputs = get_model_complexity_info(
        model,
        input_shape,
        inputs=input_constructor,
        show_table=(not args.not_show_complexity_table),
        show_arch=(not args.not_show_complexity_table))
    flops = outputs['flops_str']
    params = outputs['params_str']
    split_line = '=' * 30
    input_shape = (args.batch_size, ) + input_shape
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    if not args.not_show_complexity_table:
        print(outputs['out_table'])
        print(outputs['out_arch'])
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
