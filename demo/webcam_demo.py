# Copyright (c) OpenMMLab. All rights reserved.

import logging
from argparse import ArgumentParser

from mmcv import Config, DictAction

from mmpose.apis.webcam import WebcamExecutor


def parse_args():
    parser = ArgumentParser('Webcam executor configs')
    parser.add_argument(
        '--config', type=str, default='demo/webcam_cfg/pose_estimation.py')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='Override settings in the config. The key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options executor_cfg.camera_id=1'")
    parser.add_argument(
        '--debug', action='store_true', help='Show debug information')

    return parser.parse_args()


def run():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    webcam_exe = WebcamExecutor(**cfg.executor_cfg)
    webcam_exe.run()


if __name__ == '__main__':
    run()
