# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from datetime import datetime

import mmcv
from mmcv import Config

from mmpose.apis.inference import init_pose_model

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(
        description='calculate the parameters and flops of multiple models')

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

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.root_work_dir is None:
        # get the current time stamp
        now = datetime.now()
        ts = now.strftime('%Y_%m_%d_%H_%M')
        args.root_work_dir = f'work_dirs/flops_test_{ts}'
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

            model = init_pose_model(cfg_file)

            if hasattr(model, 'forward_dummy'):
                model.forward = model.forward_dummy
            else:
                raise NotImplementedError(
                    'FLOPs counter is currently not currently supported '
                    'with {}'.format(model.__class__.__name__))

            flops, params = get_model_complexity_info(
                model, input_shape, print_per_layer_stat=False)
            split_line = '=' * 30
            result = f'{split_line}\nModel config:{cfg_file}\n' \
                     f'Input shape: {input_shape}\n' \
                     f'Flops: {flops}\nParams: {params}\n{split_line}\n'

            print(result)
            results.append(result)

    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    with open(osp.join(args.root_work_dir, 'flops.txt'), 'w') as f:
        for res in results:
            f.write(res)


if __name__ == '__main__':
    main()
