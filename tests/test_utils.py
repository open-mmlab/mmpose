import cv2
import mmcv
import torch
import torchvision

import mmpose
from mmpose.utils import collect_env, get_root_logger


def test_collect_env():
    env_info = collect_env()

    assert env_info['PyTorch'] == torch.__version__
    assert env_info['TorchVision'] == torchvision.__version__
    assert env_info['OpenCV'] == cv2.__version__
    assert env_info['MMCV'] == mmcv.__version__
    assert env_info['MMPose'] == mmpose.__version__


def test_logger(capsys):
    logger = get_root_logger()
    logger.warning('hello')
    captured = capsys.readouterr()
    assert captured.err.endswith('mmpose - WARNING - hello\n')
