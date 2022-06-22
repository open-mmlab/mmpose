# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.models.backbones.litepose import LitePose


def test_litepose_backbone():
    model = LitePose(
        num_blocks=(6, 8, 10, 10),
        strides=(2, 2, 2, 1),
        channels=(16, 32, 48, 80),
        block_settings=(
            [[6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7]],
            [[6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7]],
            [[6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7],
             [6, 7], [6, 7]],
            [[6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7],
             [6, 7], [6, 7]],
        ),
        input_channel=16,
    )
    x = torch.randn(4, 3, 256, 256)
    outputs = model(x)
    out_channels = [16, 16, 32, 48, 80]
    sizes = [128, 64, 32, 16, 16]
    for i, output in enumerate(outputs):
        assert output.shape[1] == out_channels[i]
        assert output.shape[2] == output.shape[3] and output.shape[2] == sizes[
            i]
