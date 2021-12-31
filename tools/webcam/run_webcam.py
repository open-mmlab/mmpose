# Copyright (c) OpenMMLab. All rights reserved.

from argparse import ArgumentParser

import mmcv
from webcam_apis import WebcamRunner


def parse_args():
    parser = ArgumentParser('Lauch webcam runner')
    parser.add_argument(
        '--config',
        type=str,
        default='tools/webcam/configs/human_pose_async.py')

    return parser.parse_args()


def launch():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    runner = WebcamRunner(**cfg.runner)
    runner.run()


if __name__ == '__main__':
    launch()
