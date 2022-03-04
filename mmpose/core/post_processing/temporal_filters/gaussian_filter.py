# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d

from .builder import FILTERS
from .filter import TemporalFilter


@FILTERS.register_module(name=['GaussianFilter', 'gaussian'])
class GaussianFilter(TemporalFilter):
    """Apply median filter and then gaussian filter.

    Adapted from:
    https://github.com/akanazawa/human_dynamics/blob/mas
    ter/src/util/smooth_bbox.py.
    Args:
        x (np.ndarray): input pose
        window_size (int, optional): for median filters (must be odd).
        sigma (float, optional): Sigma for gaussian smoothing.
    Returns:
        np.ndarray: Smoothed poses
    """

    def __init__(self, window_size=11, sigma=4):
        super().__init__(window_size)
        assert window_size % 2 == 1, (
            'The window size of GaussianFilter should'
            f'be odd, but got {window_size}')
        self.sigma = sigma

    def __call__(self, x: np.ndarray):

        assert x.ndim == 3, ('Input should be an array with shape [T, K, C]'
                             f', but got invalid shape {x.shape}')

        T = x.shape[0]
        if x.shape[0] < self.window_size:
            pad_width = [(self.window_size - x.shape[0], 0), (0, 0), (0, 0)]
            x = np.pad(x, pad_width, mode='edge')

        smoothed = np.array(
            [signal.medfilt(param, self.window_size) for param in x.T]).T

        smooth_poses = np.array(
            [gaussian_filter1d(traj, self.sigma) for traj in smoothed.T]).T

        return smooth_poses[-T:]
