import warnings

import numpy as np
import scipy.signal as signal
import torch
from scipy.ndimage.filters import gaussian_filter1d

from .builder import FILTERS


@FILTERS.register_module(name=['Gaus1dFilter', 'gaus1d'])
class Gaus1dFilter:
    """Applies median filter and then gaussian filter. code from:
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
        super(Gaus1dFilter, self).__init__()

        self.window_size = window_size
        self.sigma = sigma

    def __call__(self, x=None):
        if self.window_size % 2 == 0:
            window_size = self.window_size - 1
        else:
            window_size = self.window_size
        if window_size > x.shape[0]:
            window_size = x.shape[0]
        if len(x.shape) != 3:
            warnings.warn('x should be a tensor or numpy of [T*M,K,C]')
        assert len(x.shape) == 3
        x_type = x
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu().numpy()
            else:
                x = x.numpy()

        smoothed = np.array(
            [signal.medfilt(param, window_size) for param in x.T]).T
        smooth_poses = np.array(
            [gaussian_filter1d(traj, self.sigma) for traj in smoothed.T]).T

        if isinstance(x_type, torch.Tensor):
            # we also return tensor by default
            if x_type.is_cuda:
                smooth_poses = torch.from_numpy(smooth_poses).cuda()
            else:
                smooth_poses = torch.from_numpy(smooth_poses)

        return smooth_poses
