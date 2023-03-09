# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmpose.models.backbones.tcformer import TCFormer


def test_tcformer():
    with pytest.raises(TypeError):
        # Pretrained arg must be str or None.
        TCFormer(pretrained=123)

    # test load pretrained weights
    model = TCFormer(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        num_layers=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        pretrained='https://download.openmmlab.com/mmpose/'
        'pretrain_models/tcformer-4e1adbf1_20220421.pth')
    model.init_weights()

    # test init weights from scratch
    model = TCFormer(embed_dims=[32, 32, 32, 32], num_layers=[2, 2, 2, 2])
    model.init_weights()

    # Test normal inference
    model = TCFormer()
    temp = torch.randn((1, 3, 256, 192))
    outs = model(temp)
    assert len(outs) == 4
    assert isinstance(outs[0], dict)
    for key in [
            'x', 'token_num', 'map_size', 'init_grid_size', 'idx_token',
            'agg_weight'
    ]:
        assert key in outs[0].keys()

    assert outs[0]['x'].shape == (1, 3072, 64)
    assert outs[1]['x'].shape == (1, 768, 128)
    assert outs[2]['x'].shape == (1, 192, 256)
    assert outs[3]['x'].shape == (1, 48, 512)
    assert outs[3]['idx_token'].shape == (1, 3072)
    assert outs[3]['token_num'] == 48
    assert outs[3]['map_size'] == [8, 6]
    assert outs[3]['init_grid_size'] == [64, 48]

    # Test abnormal inference size
    temp = torch.randn((1, 3, 193, 255))
    outs = model(temp)
    assert outs[0]['x'].shape == (1, 3136, 64)
    assert outs[1]['x'].shape == (1, 784, 128)
    assert outs[2]['x'].shape == (1, 196, 256)
    assert outs[3]['x'].shape == (1, 49, 512)

    # Test output feature map
    model = TCFormer(return_map=True)
    temp = torch.randn((1, 3, 256, 192))
    outs = model(temp)
    assert len(outs) == 4
    assert outs[0].shape == (1, 64, 64, 48)
    assert outs[1].shape == (1, 128, 32, 24)
    assert outs[2].shape == (1, 256, 16, 12)
    assert outs[3].shape == (1, 512, 8, 6)
