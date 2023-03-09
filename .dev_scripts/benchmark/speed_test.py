# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time
from datetime import datetime

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import IterLoader
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmpose.apis.inference import init_pose_model
from mmpose.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='test the inference speed of multiple models')

    parser.add_argument(
        '--config',
        '-c',
        help='test config file path',
        default='./.dev_scripts/benchmark/benchmark_cfg_flops_speed.yaml')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    parser.add_argument(
        '--batch-size',
        '-b',
        type=int,
        help='batch size on a single GPU',
        default=1)

    parser.add_argument(
        '--dummy-dataset-config',
        help='dummy dataset config file path',
        default='./.dev_scripts/benchmark/dummy_dataset_cfg.yaml')

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

    parser.add_argument(
        '--num-iters', type=int, help='test iterations', default=50)

    parser.add_argument(
        '--num-warmup', type=int, help='warmup iterations', default=5)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if 'cuda' in args.device.lower():
        if torch.cuda.is_available():
            with_cuda = True
        else:
            raise RuntimeError('No CUDA device found, please check it again.')
    else:
        with_cuda = False

    if args.root_work_dir is None:
        # get the current time stamp
        now = datetime.now()
        ts = now.strftime('%Y_%m_%d_%H_%M')
        args.root_work_dir = f'work_dirs/inference_speed_test_{ts}'
    mmcv.mkdir_or_exist(osp.abspath(args.root_work_dir))

    cfg = mmcv.load(args.config)
    dummy_datasets = mmcv.load(args.dummy_dataset_config)['dummy_datasets']

    results = []
    for i in range(args.priority + 1):
        models = cfg['model_list'][f'P{i}']
        for cur_model in models:
            cfg_file = cur_model['config']
            model_cfg = Config.fromfile(cfg_file)
            test_dataset = model_cfg['data']['test']
            dummy_dataset = dummy_datasets[test_dataset['type']]
            test_dataset.update(dummy_dataset)

            dataset = build_dataset(test_dataset)
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=args.batch_size,
                workers_per_gpu=model_cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False)
            data_loader = IterLoader(data_loader)

            if 'pretrained' in model_cfg.model.keys():
                del model_cfg.model['pretrained']

            model = init_pose_model(model_cfg, device=args.device.lower())

            fp16_cfg = model_cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)
            if args.fuse_conv_bn:
                model = fuse_conv_bn(model)

            # benchmark with several iterations and take the average
            pure_inf_time = 0
            speed = []
            for iteration in range(args.num_iters + args.num_warmup):
                data = next(data_loader)
                data['img'] = data['img'].to(args.device.lower())
                data['img_metas'] = data['img_metas'].data[0]

                if with_cuda:
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                with torch.no_grad():
                    model(return_loss=False, **data)

                if with_cuda:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time

                if iteration >= args.num_warmup:
                    pure_inf_time += elapsed
                    speed.append(1 / elapsed)

            speed_mean = np.mean(speed)
            speed_std = np.std(speed)

            split_line = '=' * 30
            result = f'{split_line}\nModel config:{cfg_file}\n' \
                     f'Device: {args.device}\n' \
                     f'Batch size: {args.batch_size}\n' \
                     f'Overall average speed: {speed_mean:.2f} \u00B1 ' \
                     f'{speed_std:.2f} items / s\n' \
                     f'Total iters: {args.num_iters}\n'\
                     f'Total time: {pure_inf_time:.2f} s \n{split_line}\n'\

            print(result)
            results.append(result)

    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are included and verify that the '
          'speed computation is correct.')
    with open(osp.join(args.root_work_dir, 'inference_speed.txt'), 'w') as f:
        for res in results:
            f.write(res)


if __name__ == '__main__':
    main()
