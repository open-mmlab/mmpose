# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine.config import Config, DictAction

from mmpose.registry import MODELS
from mmpose.utils import register_all_modules

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    register_all_modules()
    args = parse_args()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    input_shape = (3, h, w)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = MODELS.build(cfg.model)
    model.eval()

    analysis_results = get_model_complexity_info(
        model, input_shape, show_table=True, show_arch=False)

    # ayalysis_results = {
    #     'flops': flops,
    #     'flops_str': flops_str,
    #     'activations': activations,
    #     'activations_str': activations_str,
    #     'params': params,
    #     'params_str': params_str,
    #     'out_table': complexity_table,
    #     'out_arch': complexity_arch
    # }

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {analysis_results["flops"]}\n'
          f'Params: {analysis_results["params"]}\n{split_line}')

    print(analysis_results['activations'])
    # print(analysis_results['complexity_table'])
    # print(complexity_str)
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
