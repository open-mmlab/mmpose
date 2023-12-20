# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList
from mmengine.utils import digit_version, to_2tuple
from mmengine.utils.dl_utils import TORCH_VERSION
from torch import Tensor

from mmpose.utils.typing import ConfigType, OptConfigType

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        """Get horizontal and vertical padding shapes."""

        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        """Forward function."""

        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        conv_type='Conv2d',
        kernel_size=16,
        stride=16,
        padding='corner',
        dilation=1,
        bias=True,
        norm_cfg=None,
        input_size=None,
        init_cfg=None,
    ):
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified..
            Default: True.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


class ScaleNorm(nn.Module):
    """Scale Norm.

    Args:
        dim (int): The dimension of the scale vector.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.

    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`_
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The tensor after applying scale norm.
        """

        if torch.onnx.is_in_onnx_export() and \
                digit_version(TORCH_VERSION) >= digit_version('1.12'):

            norm = torch.linalg.norm(x, dim=-1, keepdim=True)

        else:
            norm = torch.norm(x, dim=-1, keepdim=True)
        norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class SinePositionalEncoding(nn.Module):
    """Sine Positional Encoding Module. This module implements sine positional
    encoding, which is commonly used in transformer-based models to add
    positional information to the input sequences. It uses sine and cosine
    functions to create positional embeddings for each element in the input
    sequence.

    Args:
        out_channels (int): The number of features in the input sequence.
        temperature (int): A temperature parameter used to scale
            the positional encodings. Defaults to 10000.
        spatial_dim (int): The number of spatial dimension of input
            feature. 1 represents sequence data and 2 represents grid data.
            Defaults to 1.
        learnable (bool): Whether to optimize the frequency base. Defaults
            to False.
        eval_size (int, tuple[int], optional): The fixed spatial size of
            input features. Defaults to None.
    """

    def __init__(
        self,
        out_channels: int,
        spatial_dim: int = 1,
        temperature: int = 1e5,
        learnable: bool = False,
        eval_size: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:

        super().__init__()

        assert out_channels % 2 == 0
        assert temperature > 0

        self.spatial_dim = spatial_dim
        self.out_channels = out_channels
        self.temperature = temperature
        self.eval_size = eval_size
        self.learnable = learnable

        pos_dim = out_channels // 2
        dim_t = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        dim_t = self.temperature**(dim_t)

        if not learnable:
            self.register_buffer('dim_t', dim_t)
        else:
            self.dim_t = nn.Parameter(dim_t.detach())

        # set parameters
        if eval_size:
            if hasattr(self, f'pos_enc_{eval_size}'):
                delattr(self, f'pos_enc_{eval_size}')
            pos_enc = self.generate_pos_encoding(size=eval_size)
            self.register_buffer(f'pos_enc_{eval_size}', pos_enc)

    def forward(self, *args, **kwargs):
        return self.generate_pos_encoding(*args, **kwargs)

    def generate_pos_encoding(self,
                              size: Union[int, Sequence[int]] = None,
                              position: Optional[Tensor] = None):
        """Generate positional encoding for input features.

        Args:
            size (int or tuple[int]): Size of the input features. Required
                if position is None.
            position (Tensor, optional): Position tensor. Required if size
                is None.
        """

        assert (size is not None) ^ (position is not None)

        if (not (self.learnable
                 and self.training)) and size is not None and hasattr(
                     self, f'pos_enc_{size}'):
            return getattr(self, f'pos_enc_{size}')

        if self.spatial_dim == 1:
            if size is not None:
                if isinstance(size, (tuple, list)):
                    size = size[0]
                position = torch.arange(
                    size, dtype=torch.float32, device=self.dim_t.device)

            dim_t = self.dim_t.reshape(*((1, ) * position.ndim), -1)
            freq = position.unsqueeze(-1) / dim_t
            pos_enc = torch.cat((freq.cos(), freq.sin()), dim=-1)

        elif self.spatial_dim == 2:
            if size is not None:
                if isinstance(size, (tuple, list)):
                    h, w = size[:2]
                elif isinstance(size, (int, float)):
                    h, w = int(size), int(size)
                else:
                    raise ValueError(f'got invalid type {type(size)} for size')
                grid_h, grid_w = torch.meshgrid(
                    torch.arange(
                        int(h), dtype=torch.float32, device=self.dim_t.device),
                    torch.arange(
                        int(w), dtype=torch.float32, device=self.dim_t.device))
                grid_h, grid_w = grid_h.flatten(), grid_w.flatten()
            else:
                assert position.size(-1) == 2
                grid_h, grid_w = torch.unbind(position, dim=-1)

            dim_t = self.dim_t.reshape(*((1, ) * grid_h.ndim), -1)
            freq_h = grid_h.unsqueeze(-1) / dim_t
            freq_w = grid_w.unsqueeze(-1) / dim_t
            pos_enc_h = torch.cat((freq_h.cos(), freq_h.sin()), dim=-1)
            pos_enc_w = torch.cat((freq_w.cos(), freq_w.sin()), dim=-1)
            pos_enc = torch.stack((pos_enc_h, pos_enc_w), dim=-1)

        return pos_enc

    @staticmethod
    def apply_additional_pos_enc(feature: Tensor,
                                 pos_enc: Tensor,
                                 spatial_dim: int = 1):
        """Apply additional positional encoding to input features.

        Args:
            feature (Tensor): Input feature tensor.
            pos_enc (Tensor): Positional encoding tensor.
            spatial_dim (int): Spatial dimension of input features.
        """

        assert spatial_dim in (1, 2), f'the argument spatial_dim must be ' \
            f'either 1 or 2, but got {spatial_dim}'
        if spatial_dim == 2:
            pos_enc = pos_enc.flatten(-2)
        for _ in range(feature.ndim - pos_enc.ndim):
            pos_enc = pos_enc.unsqueeze(0)
        return feature + pos_enc

    @staticmethod
    def apply_rotary_pos_enc(feature: Tensor,
                             pos_enc: Tensor,
                             spatial_dim: int = 1):
        """Apply rotary positional encoding to input features.

        Args:
            feature (Tensor): Input feature tensor.
            pos_enc (Tensor): Positional encoding tensor.
            spatial_dim (int): Spatial dimension of input features.
        """

        assert spatial_dim in (1, 2), f'the argument spatial_dim must be ' \
            f'either 1 or 2, but got {spatial_dim}'

        for _ in range(feature.ndim - pos_enc.ndim + spatial_dim - 1):
            pos_enc = pos_enc.unsqueeze(0)

        x1, x2 = torch.chunk(feature, 2, dim=-1)
        if spatial_dim == 1:
            cos, sin = torch.chunk(pos_enc, 2, dim=-1)
            feature = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin),
                                dim=-1)
        elif spatial_dim == 2:
            pos_enc_h, pos_enc_w = torch.unbind(pos_enc, dim=-1)
            cos_h, sin_h = torch.chunk(pos_enc_h, 2, dim=-1)
            cos_w, sin_w = torch.chunk(pos_enc_w, 2, dim=-1)
            feature = torch.cat(
                (x1 * cos_h - x2 * sin_h, x1 * cos_w + x2 * sin_w), dim=-1)

        return feature


class ChannelWiseScale(nn.Module):
    """Scale vector by element multiplications.

    Args:
        dim (int): The dimension of the scale vector.
        init_value (float, optional): The initial value of the scale vector.
            Defaults to 1.0.
        trainable (bool, optional): Whether the scale vector is trainable.
            Defaults to True.
    """

    def __init__(self, dim, init_value=1., trainable=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        """Forward function."""

        return x * self.scale


class GAUEncoder(BaseModule):
    """Gated Attention Unit (GAU) Encoder.

    Args:
        in_token_dims (int): The input token dimension.
        out_token_dims (int): The output token dimension.
        expansion_factor (int, optional): The expansion factor of the
            intermediate token dimension. Defaults to 2.
        s (int, optional): The self-attention feature dimension.
            Defaults to 128.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
        drop_path (float, optional): The drop path rate. Defaults to 0.0.
        act_fn (str, optional): The activation function which should be one
            of the following options:

            - 'ReLU': ReLU activation.
            - 'SiLU': SiLU activation.

            Defaults to 'SiLU'.
        bias (bool, optional): Whether to use bias in linear layers.
            Defaults to False.
        pos_enc (bool, optional): Whether to use rotary position
            embedding. Defaults to False.
        spatial_dim (int, optional): The spatial dimension of inputs

    Reference:
        `Transformer Quality in Linear Time
        <https://arxiv.org/abs/2202.10447>`_
    """

    def __init__(self,
                 in_token_dims,
                 out_token_dims,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5,
                 dropout_rate=0.,
                 drop_path=0.,
                 act_fn='SiLU',
                 bias=False,
                 pos_enc: str = 'none',
                 spatial_dim: int = 1):

        super(GAUEncoder, self).__init__()
        self.s = s
        self.bias = bias
        self.pos_enc = pos_enc
        self.in_token_dims = in_token_dims
        self.spatial_dim = spatial_dim
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        self._build_layers()

        self.ln = ScaleNorm(in_token_dims, eps=eps)

        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn == 'SiLU':
            assert digit_version(TORCH_VERSION) >= digit_version('1.7.0'), \
                'SiLU activation requires PyTorch version >= 1.7'

            self.act_fn = nn.SiLU(True)
        else:
            self.act_fn = nn.ReLU(True)

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = ChannelWiseScale(in_token_dims)
        else:
            self.shortcut = False

        self.sqrt_s = math.sqrt(s)
        self.dropout_rate = dropout_rate

        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

    def _build_layers(self):
        self.uv = nn.Linear(
            self.in_token_dims, 2 * self.e + self.s, bias=self.bias)
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))

    def _forward(self, x, mask=None, pos_enc=None):
        """GAU Forward function."""

        x = self.ln(x)

        # [B, K, in_token_dims] -> [B, K, e + e + s]
        uv = self.uv(x)
        uv = self.act_fn(uv)

        # [B, K, e + e + s] -> [B, K, e], [B, K, e], [B, K, s]
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)
        # [B, K, 1, s] * [1, 1, 2, s] + [2, s] -> [B, K, 2, s]
        dim = base.ndim - self.gamma.ndim + 1
        gamma = self.gamma.view(*((1, ) * dim), *self.gamma.size())
        beta = self.beta.view(*((1, ) * dim), *self.beta.size())
        base = base.unsqueeze(-2) * gamma + beta
        # [B, K, 2, s] -> [B, K, s], [B, K, s]
        q, k = torch.unbind(base, dim=-2)

        if self.pos_enc == 'rope':
            q = SinePositionalEncoding.apply_rotary_pos_enc(
                q, pos_enc, self.spatial_dim)
            k = SinePositionalEncoding.apply_rotary_pos_enc(
                k, pos_enc, self.spatial_dim)
        elif self.pos_enc == 'add':
            pos_enc = pos_enc.reshape(*((1, ) * (q.ndim - 2)), q.size(-2),
                                      q.size(-1))
            q = q + pos_enc
            k = k + pos_enc

        # [B, K, s].transpose(-1, -2) -> [B, s, K]
        # [B, K, s] x [B, s, K] -> [B, K, K]
        qk = torch.matmul(q, k.transpose(-1, -2))

        # [B, K, K]
        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if mask is not None:
            kernel = kernel * mask

        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)

        # [B, K, K] x [B, K, e] -> [B, K, e]
        x = u * torch.matmul(kernel, v)
        # [B, K, e] -> [B, K, out_token_dims]
        x = self.o(x)

        return x

    def forward(self, x, mask=None, pos_enc=None):
        """Forward function."""
        out = self.drop_path(self._forward(x, mask=mask, pos_enc=pos_enc))
        if self.shortcut:
            return self.res_scale(x) + out
        else:
            return out


class DetrTransformerEncoder(BaseModule):
    """Encoder of DETR.

    Args:
        num_layers (int): Number of encoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        num_cp (int): Number of checkpointing blocks in encoder layer.
            Default to -1.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 num_cp: int = -1,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of encoder.

        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).

        Returns:
            Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
            `True`, otherwise (num_queries, bs, dim).
        """
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask, **kwargs)
        return query


class DetrTransformerEncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query
