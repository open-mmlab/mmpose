# Copyright (c) OpenMMLab. All rights reserved.

from argparse import ArgumentParser

from mmcv import Config, DictAction
from webcam_apis import WebcamRunner


def parse_args():
    parser = ArgumentParser('Lauch webcam runner')
    parser.add_argument(
        '--config',
        type=str,
        default='tools/webcam/configs/meow_dwen_dwen/meow_dwen_dwen.py')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options runner.camera_id=1 runner.synchronous=True'")

    return parser.parse_args()


def launch():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    runner = WebcamRunner(**cfg.runner)
    runner.run()


if __name__ == '__main__':
    launch()
