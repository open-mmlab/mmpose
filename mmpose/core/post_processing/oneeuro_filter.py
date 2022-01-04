import math
import warnings

import numpy as np
import torch

from .builder import FILTERS


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuro:

    def __init__(self,
                 t0,
                 x0,
                 dx0=0.0,
                 min_cutoff=1.0,
                 beta=0.0,
                 d_cutoff=1.0):
        super(OneEuro, self).__init__()
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)  # [k, c]
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)
        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


@FILTERS.register_module(name=['OneEuroFilter', 'oneeuro'])
class OneEuroFilter:
    """Oneeuro filter, source code: https://github.com/mkocabas/VIBE/blob/c0
    c3f77d587351c806e901221a9dc05d1ffade4b/lib/utils/smooth_pose.py.
    Args:
        min_cutoff (float, optional):
        Decreasing the minimum cutoff frequency decreases slow speed jitter
        beta (float, optional):
        Increasing the speed coefficient(beta) decreases speed lag.
    Returns:
        np.ndarray: smoothed poses
    """

    def __init__(self, min_cutoff=0.004, beta=0.7):
        super(OneEuroFilter, self).__init__()

        self.min_cutoff = min_cutoff
        self.beta = beta

    def __call__(self, x=None):
        # x (np.ndarray): input poses.
        if len(x.shape) != 3:
            warnings.warn('x should be a tensor or numpy of [T*M,K,C]')
        assert len(x.shape) == 3
        x_type = x
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu().numpy()
            else:
                x = x.numpy()

        one_euro_filter = OneEuro(
            np.zeros_like(x[0]),
            x[0],
            min_cutoff=self.min_cutoff,
            beta=self.beta,
        )

        pred_pose_hat = np.zeros_like(x)

        # initialize
        pred_pose_hat[0] = x[0]

        for idx, pose in enumerate(x[1:]):
            idx += 1
            t = np.ones_like(pose) * idx
            pose = one_euro_filter(t, pose)
            pred_pose_hat[idx] = pose

        if isinstance(x_type, torch.Tensor):
            # we also return tensor by default
            if x_type.is_cuda:
                pred_pose_hat = torch.from_numpy(pred_pose_hat).cuda()
            else:
                pred_pose_hat = torch.from_numpy(pred_pose_hat)
        return pred_pose_hat
