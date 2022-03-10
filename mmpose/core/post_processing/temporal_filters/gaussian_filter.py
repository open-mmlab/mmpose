# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import medfilt

from .builder import FILTERS
from .filter import TemporalFilter


@FILTERS.register_module(name=['GaussianFilter', 'gaussian'])
class GaussianFilter(TemporalFilter):
    """Apply median filter and then gaussian filter.

    Adapted from:
    https://github.com/akanazawa/human_dynamics/blob/mas
    ter/src/util/smooth_bbox.py.

    Args:
        window_size (int): The size of the filter window (i.e., the number
            of coefficients). window_length must be a positive odd integer.
            Default: 11
        sigma (float): Sigma for gaussian smoothing. Default: 4.0
    """

    def __init__(self, window_size: int = 11, sigma: float = 4.0):
        super().__init__(window_size)
        assert window_size % 2 == 1, (
            'The window size of GaussianFilter should'
            f'be odd, but got {window_size}')
        self.sigma = sigma

    def __call__(self, x: np.ndarray):

        assert x.ndim == 3, ('Input should be an array with shape [T, K, C]'
                             f', but got invalid shape {x.shape}')

        if x.shape[0] < self.window_size:
            pad_width = [(self.window_size - x.shape[0], 0), (0, 0), (0, 0)]
            x = np.pad(x, pad_width, mode='edge')
        y = medfilt(x, (self.window_size, 1, 1))
        y = gaussian_filter1d(y, self.sigma, axis=0)
        return y
