import cv2
import mmcv
import torch
import torchvision

import mmpose
from mmpose.utils import collect_env


def test_collect_env():
    env_info = collect_env()
    assert env_info['PyTorch'] == torch.__version__
    assert env_info['TorchVision'] == torchvision.__version__
    assert env_info['OpenCV'] == cv2.__version__
    assert env_info['MMCV'] == mmcv.__version__
    assert '+' in env_info['MMPose']
    assert mmpose.__version__ in env_info['MMPose']
