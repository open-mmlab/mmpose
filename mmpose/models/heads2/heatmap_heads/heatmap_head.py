# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.core.utils.typing import ConfigType, OptConfigType
from mmpose.registry import MODELS
from ..base_head import BaseHead


@MODELS.register_module()
class HeatmapHead(BaseHead):
    """Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
    (2018). The head is composed of a few deconvolutional layers followed by a
    convolutional layer to generate heatmaps from low-resolution feature maps.

    Args:
        in_channels (int): Number of channels in the input feature map
        heatmap_channels (int): Number of channels in the heatmap, which
            should equals to the keypoint number
        num_deconv_layers (int): Number of deconv layers. Defaults to 3
        deconv_out_channels (sequence[int], optional): The output channel
            number of each deconv layer. The length should equal to
            ``num_deconv_layers``. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (sequence[int | tuple]): The kernel size of
            each deconv layer. Each element should be either a integer
        input_transform (str): Transformation of input features which should
            be one of the following options:

                - ``'resize_concat'``: Resize multiple feature maps specified
                    by ``input_index`` to the same size as the first one and
                    concat these feature maps
                - ``'select'``: Select feature map(s) specified by
                    ``input_index``. Multiple selected features will be
                    bundled into a tuple
            Defaults to ``'select'``
        input_index (int | sequence[int]): The feature map index used in the
            input transformation. See also ``input_transform``. Defaults to 0
        align_corners (bool): `align_corners` argument of
            :func:`torch.nn.functional.interpolate` used in the input
            transformation. Defaults to ``False``
        loss (Config): Config of the keypoint loss
        init_cfg (Config, optional):

    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    def __init__(self, loss: ConfigType = {}, init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
