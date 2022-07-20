# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing as mp
import os
import platform
import time

import cv2
import mmcv
import torch
import torchvision
from mmcv import Config

import mmpose
from mmpose.utils import StopWatch, collect_env, setup_multi_processes


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

    _ = stop_watch.report()
    _ = stop_watch.report_strings()


def test_setup_multi_processes():
    # temp save system setting
    sys_start_mehod = mp.get_start_method(allow_none=True)
    sys_cv_threads = cv2.getNumThreads()
    # pop and temp save system env vars
    sys_omp_threads = os.environ.pop('OMP_NUM_THREADS', default=None)
    sys_mkl_threads = os.environ.pop('MKL_NUM_THREADS', default=None)

    # test config without setting env
    config = dict(data=dict(workers_per_gpu=2))
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert os.getenv('OMP_NUM_THREADS') == '1'
    assert os.getenv('MKL_NUM_THREADS') == '1'
    # when set to 0, the num threads will be 1
    assert cv2.getNumThreads() == 1
    if platform.system() != 'Windows':
        assert mp.get_start_method() == 'fork'

    # test num workers <= 1
    os.environ.pop('OMP_NUM_THREADS')
    os.environ.pop('MKL_NUM_THREADS')
    config = dict(data=dict(workers_per_gpu=0))
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert 'OMP_NUM_THREADS' not in os.environ
    assert 'MKL_NUM_THREADS' not in os.environ

    # test manually set env var
    os.environ['OMP_NUM_THREADS'] = '4'
    config = dict(data=dict(workers_per_gpu=2))
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert os.getenv('OMP_NUM_THREADS') == '4'

    # test manually set opencv threads and mp start method
    config = dict(
        data=dict(workers_per_gpu=2),
        opencv_num_threads=4,
        mp_start_method='spawn')
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert cv2.getNumThreads() == 4
    assert mp.get_start_method() == 'spawn'

    # revert setting to avoid affecting other programs
    if sys_start_mehod:
        mp.set_start_method(sys_start_mehod, force=True)
    cv2.setNumThreads(sys_cv_threads)
    if sys_omp_threads:
        os.environ['OMP_NUM_THREADS'] = sys_omp_threads
    else:
        os.environ.pop('OMP_NUM_THREADS')
    if sys_mkl_threads:
        os.environ['MKL_NUM_THREADS'] = sys_mkl_threads
    else:
        os.environ.pop('MKL_NUM_THREADS')
