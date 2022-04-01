# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from torch import Tensor, nn

from .builder import FILTERS
from .filter import TemporalFilter


class SmoothNetResBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out


class SmoothNet(nn.Module):
    """SmoothNet is a plug-and-play temporal-only network to refine human
    poses. It works for 2d/3d/6d pose smoothing.

    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .

    Note:
        N: The batch size
        T: The temporal length of the pose sequence
        C: The total pose dimension (e.g. keypoint_number * keypoint_dim)

    Args:
        window_size (int): The size of the input window.
        output_size (int): The size of the output window.
        hidden_size (int): The hidden feature dimension in the encoder,
            the decoder and between residual blocks. Default: 512
        res_hidden_size (int): The hidden feature dimension inside the
            residual blocks. Default: 256
        num_blocks (int): The number of residual blocks. Default: 3
        dropout (float): Dropout probability. Default: 0.5

    Shape:
        Input: (N, C, T) the original pose sequence
        Output: (N, C, T) the smoothed pose sequence
    """

    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout

        assert output_size <= window_size, (
            'The output size should be less than or equal to the window size.',
            f' Got output_size=={output_size} and window_size=={window_size}')

        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        # Build residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(
                SmoothNetResBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Build decoder layers
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        N, C, T = x.shape
        num_windows = T - self.window_size + 1

        assert T >= self.window_size, (
            'Input sequence length must be no less than the window size. ',
            f'Got x.shape[2]=={T} and window_size=={self.window_size}')

        # Unfold x to obtain input sliding windows
        # [N, C, num_windows, window_size]
        x = x.unfold(2, self.window_size, 1)

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, num_windows, output_size]

        # Accumulate output ensembles
        out = x.new_zeros(N, C, T)
        count = x.new_zeros(T)

        for t in range(num_windows):
            out[..., t:t + self.output_size] += x[:, :, t]
            count[t:t + self.output_size] += 1.0

        return out.div(count)


@FILTERS.register_module(name=['SmoothNet', 'Smoothnet', 'smoothnet'])
class SmoothNetFilter(TemporalFilter):
    """Apply SmoothNet filter.

    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .

    Args:
        window_size (int): The size of the filter window (i.e., the number
            of coefficients). window_length must be a positive odd integer.
        checkpoint (str): The checkpoint file of the pretrained SmoothNet
            model. Please note that `checkpoint` and `window_size` should
            be matched.
        device (str)
    """

    def __init__(
        self,
        window_size: int,
        checkpoint: str,
        output_size,
        hidden_size: int = 512,
        res_hidden_size: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.5,
        device: str = 'cpu',
    ):
        super().__init__(window_size)
        self.device = device
        self.smoothnet = SmoothNet(window_size, output_size, hidden_size,
                                   res_hidden_size, num_blocks, dropout)
        load_checkpoint(self.smoothnet, checkpoint)
        self.smoothnet.to(device)
        self.smoothnet.eval()

    def __call__(self, x: np.ndarray):

        assert x.ndim == 3, ('Input should be an array with shape [T, K, C]'
                             f', but got invalid shape {x.shape}')

        T, K, C = x.shape
        # Do not apply the filter if the input length is less than the window
        # size.
        if T < self.window_size:
            return x

        dtype = x.dtype

        # Convert to tensor and forward the model
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = x.view(1, T, K * C).permute(0, 2, 1)  # to [1, KC, T]
        smoothed = self.smoothnet(x)  # in shape [1, KC, T]

        # Convert model output back to input shape and format
        smoothed = smoothed.permute(0, 2, 1).view(T, K, C)  # to [T, K, C]
        smoothed = smoothed.cpu().numpy().astype(dtype)  # to numpy.ndarray

        return smoothed
