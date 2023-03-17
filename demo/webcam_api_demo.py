# Copyright (c) OpenMMLab. All rights reserved.

import logging
from argparse import ArgumentParser

from mmengine import Config, DictAction

from mmpose.apis.webcam import WebcamExecutor
from mmpose.apis.webcam.nodes import model_nodes


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
        '--debug', action='store_true', help='Show debug information.')
    parser.add_argument(
        '--cpu', action='store_true', help='Use CPU for model inference.')
    parser.add_argument(
        '--cuda', action='store_true', help='Use GPU for model inference.')

    return parser.parse_args()


def set_device(cfg: Config, device: str):
    """Set model device in config.

    Args:
        cfg (Config): Webcam config
        device (str): device indicator like "cpu" or "cuda:0"
    """

    device = device.lower()
    assert device == 'cpu' or device.startswith('cuda:')

    for node_cfg in cfg.executor_cfg.nodes:
        if node_cfg.type in model_nodes.__all__:
            node_cfg.update(device=device)

    return cfg


def run():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.cpu:
        cfg = set_device(cfg, 'cpu')

    if args.cuda:
        cfg = set_device(cfg, 'cuda:0')

    webcam_exe = WebcamExecutor(**cfg.executor_cfg)
    webcam_exe.run()


if __name__ == '__main__':
    run()
