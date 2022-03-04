# ------------------------------------------------------------------------------
# Adapted from https://github.com/HoBeom/OneEuroFilter-Numpy
# Original licence: Copyright (c)  HoBeom Jeon, under the MIT License.
# ------------------------------------------------------------------------------
import math

import numpy as np

from .builder import FILTERS
from .filter import TemporalFilter


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuro:

    def __init__(self, t0, x0, dx0, min_cutoff, beta, d_cutoff=1.0):
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
class OneEuroFilter(TemporalFilter):
    """Oneeuro filter, source code: https://github.com/mkocabas/VIBE/blob/c0
    c3f77d587351c806e901221a9dc05d1ffade4b/lib/utils/smooth_pose.py.

    Args:
        min_cutoff (float, optional): Decreasing the minimum cutoff frequency
            decreases slow speed jitter
        beta (float, optional): Increasing the speed coefficient(beta)
            decreases speed lag.
    Returns:
        np.ndarray: smoothed poses
    """

    def __init__(self, min_cutoff=0.004, beta=0.7):
        # OneEuroFilter has Markov Property and maintains status variables
        # within the class, thus has a windows_size of 1
        super().__init__(window_size=1)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._one_euro = None

    def __call__(self, x=None):
        assert x.ndim == 3, ('Input should be an array with shape [T, K, C]'
                             f', but got invalid shape {x.shape}')

        if self._one_euro is None:
            self._one_euro = OneEuro(
                np.zeros_like(x[0]),
                x[0],
                dx0=0.0,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
            )

        pred_pose_hat = np.zeros_like(x)

        # initialize
        pred_pose_hat[0] = x[0]

        for idx, pose in enumerate(x[1:]):
            idx += 1
            t = np.ones_like(pose) * idx
            pose = self._one_euro(t, pose)
            pred_pose_hat[idx] = pose

        return pred_pose_hat
