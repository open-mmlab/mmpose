# Copyright (c) OpenMMLab. All rights reserved.
import time

import cv2
import mmcv
import numpy as np
import torch
import torchvision

import mmpose
from mmpose.utils import StopWatch, collect_env


def test_collect_env():
    env_info = collect_env()
    assert env_info['PyTorch'] == torch.__version__
    assert env_info['TorchVision'] == torchvision.__version__
    assert env_info['OpenCV'] == cv2.__version__
    assert env_info['MMCV'] == mmcv.__version__
    assert '+' in env_info['MMPose']
    assert mmpose.__version__ in env_info['MMPose']


def test_stopwatch():
    window_size = 5
    test_loop = 10
    outer_time = 100
    inner_time = 100

    stop_watch = StopWatch(window=window_size)
    for _ in range(test_loop):
        with stop_watch.timeit():
            time.sleep(outer_time / 1000.)
            with stop_watch.timeit('inner'):
                time.sleep(inner_time / 1000.)

    report = stop_watch.report()
    _ = stop_watch.report_strings()

    np.testing.assert_allclose(
        report['_FPS_'], outer_time + inner_time, rtol=0.01)

    np.testing.assert_allclose(report['inner'], inner_time, rtol=0.01)
