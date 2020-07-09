import os.path as osp
import subprocess
import sys
from collections import defaultdict

import cv2
import mmcv
import torch
import torchvision
from mmcv.utils import CUDA_HOME, get_build_config

import mmpose
from mmpose.utils import collect_env


def test_collect_env():
    env_info = collect_env()
    target_keys = [
        'sys.platform', 'Python', 'CUDA available', 'GCC', 'PyTorch',
        'PyTorch compiling details', 'TorchVision', 'OpenCV', 'MMCV', 'MMPose'
    ]
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_keys = ['CUDA_HOME', 'NVCC']
        devices = defaultdict(list)
        devices_dict = dict()
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            devices_dict['GPU ' + ','.join(devids)] = name
            cuda_keys.append('GPU ' + ','.join(devids))
        target_keys.extend(cuda_keys)

    assert set(env_info.keys()) == set(target_keys)
    assert env_info['sys.platform'] == sys.platform
    assert env_info['Python'] == sys.version.replace('\n', '')
    assert env_info['CUDA available'] == cuda_available
    if cuda_available:
        assert env_info['CUDA_HOME'] == CUDA_HOME
        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    '"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            assert env_info['NVCC'] == nvcc

        for k, v in devices_dict.items():
            assert env_info[k] == v

    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    assert env_info['GCC'] == gcc

    assert env_info['PyTorch'] == torch.__version__
    assert env_info['PyTorch compiling details'] == get_build_config()

    assert env_info['TorchVision'] == torchvision.__version__

    assert env_info['OpenCV'] == cv2.__version__

    assert env_info['MMCV'] == mmcv.__version__
    assert env_info['MMPose'] == mmpose.__version__
