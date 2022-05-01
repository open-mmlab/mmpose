# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmpose.models.backbones.pvt import (PVTEncoderLayer,
                                         PyramidVisionTransformer,
                                         PyramidVisionTransformerV2)


def test_pvt_block():
    # test PVT structure and forward
    block = PVTEncoderLayer(
        embed_dims=64, num_heads=4, feedforward_channels=256)
    assert block.ffn.embed_dims == 64
    assert block.attn.num_heads == 4
    assert block.ffn.feedforward_channels == 256
    x = torch.randn(1, 56 * 56, 64)
    x_out = block(x, (56, 56))
    assert x_out.shape == torch.Size([1, 56 * 56, 64])


def test_pvt():
    """Test PVT backbone."""

    with pytest.raises(TypeError):
        # Pretrained arg must be str or None.
        PyramidVisionTransformer(pretrained=123)

    # test pretrained image size
    with pytest.raises(AssertionError):
        PyramidVisionTransformer(pretrain_img_size=(224, 224, 224))

    # test padding
    model = PyramidVisionTransformer(
        paddings=['corner', 'corner', 'corner', 'corner'])
    temp = torch.randn((1, 3, 32, 32))
    outs = model(temp)
    assert outs[0].shape == (1, 64, 8, 8)
    assert outs[1].shape == (1, 128, 4, 4)
    assert outs[2].shape == (1, 320, 2, 2)
    assert outs[3].shape == (1, 512, 1, 1)

    # Test absolute position embedding
    temp = torch.randn((1, 3, 224, 224))
    model = PyramidVisionTransformer(
        pretrain_img_size=224, use_abs_pos_embed=True)
    model.init_weights()
    model(temp)

    # Test normal inference
    temp = torch.randn((1, 3, 32, 32))
    model = PyramidVisionTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 8, 8)
    assert outs[1].shape == (1, 128, 4, 4)
    assert outs[2].shape == (1, 320, 2, 2)
    assert outs[3].shape == (1, 512, 1, 1)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 33, 33))
    model = PyramidVisionTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 8, 8)
    assert outs[1].shape == (1, 128, 4, 4)
    assert outs[2].shape == (1, 320, 2, 2)
    assert outs[3].shape == (1, 512, 1, 1)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 112, 137))
    model = PyramidVisionTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 28, 34)
    assert outs[1].shape == (1, 128, 14, 17)
    assert outs[2].shape == (1, 320, 7, 8)
    assert outs[3].shape == (1, 512, 3, 4)


def test_pvtv2():
    """Test PVTv2 backbone."""

    with pytest.raises(TypeError):
        # Pretrained arg must be str or None.
        PyramidVisionTransformerV2(pretrained=123)

    # test pretrained image size
    with pytest.raises(AssertionError):
        PyramidVisionTransformerV2(pretrain_img_size=(224, 224, 224))

    # test load pretrained weights
    model = PyramidVisionTransformerV2(
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        pretrained='https://github.com/whai362/PVT/'
        'releases/download/v2/pvt_v2_b0.pth')
    model.init_weights()

    # test init_cfg
    model = PyramidVisionTransformerV2(
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b0.pth'))
    model.init_weights()

    # test init weights from scratch
    model = PyramidVisionTransformerV2(embed_dims=32, num_layers=[2, 2, 2, 2])
    model.init_weights()

    # Test normal inference
    temp = torch.randn((1, 3, 32, 32))
    model = PyramidVisionTransformerV2()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 8, 8)
    assert outs[1].shape == (1, 128, 4, 4)
    assert outs[2].shape == (1, 320, 2, 2)
    assert outs[3].shape == (1, 512, 1, 1)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 31, 31))
    model = PyramidVisionTransformerV2()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 8, 8)
    assert outs[1].shape == (1, 128, 4, 4)
    assert outs[2].shape == (1, 320, 2, 2)
    assert outs[3].shape == (1, 512, 1, 1)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 112, 137))
    model = PyramidVisionTransformerV2()
    outs = model(temp)
    assert outs[0].shape == (1, 64, 28, 35)
    assert outs[1].shape == (1, 128, 14, 18)
    assert outs[2].shape == (1, 320, 7, 9)
    assert outs[3].shape == (1, 512, 4, 5)
