# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones.tcformer import TCFormer
from mmpose.models.necks.tcformer_mta_neck import MTA


def test_mta():
    in_channels = [8, 16, 32, 64]
    out_channels = 8

    # end_level=-1 is equal to end_level=3
    MTA(in_channels=in_channels,
        out_channels=out_channels,
        start_level=0,
        end_level=-1,
        num_outs=5)
    MTA(in_channels=in_channels,
        out_channels=out_channels,
        start_level=0,
        end_level=3,
        num_outs=5)

    # `num_outs` is not equal to end_level - start_level + 1
    with pytest.raises(AssertionError):
        MTA(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=2,
            num_outs=3)

    # `num_outs` SMALLER  len(in_channels) - start_level
    with pytest.raises(AssertionError):
        MTA(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            num_outs=2)

    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        MTA(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=4,
            num_outs=2)

    # `num_outs` is not equal to end_level - start_level
    with pytest.raises(AssertionError):
        MTA(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=3,
            num_outs=1)

    # Invalid `add_extra_convs` option
    with pytest.raises(AssertionError):
        MTA(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            add_extra_convs='on_xxx',
            num_outs=5)

    backbone = TCFormer(embed_dims=[8, 16, 32, 64])
    temp = torch.randn((1, 3, 256, 192))
    feats = backbone(temp)
    h, w = 64, 48

    # normal forward
    mta_model = MTA(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs=True,
        num_outs=5)
    assert mta_model.add_extra_convs == 'on_input'
    outs = mta_model(feats)
    assert len(outs) == 5
    for i in range(mta_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == h // (2**i)
        outs[i].shape[3] == w // (2**i)

    # Tests for mta with no extra convs (pooling is used instead)
    mta_model = MTA(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=False,
        num_outs=5)
    outs = mta_model(feats)
    assert len(outs) == mta_model.num_outs
    assert not mta_model.add_extra_convs
    for i in range(mta_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == h // (2**i)
        outs[i].shape[3] == w // (2**i)

    # Tests for mta with lateral bns
    mta_model = MTA(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        no_norm_on_lateral=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        num_outs=5)
    outs = mta_model(feats)
    assert len(outs) == mta_model.num_outs
    assert mta_model.add_extra_convs == 'on_input'
    for i in range(mta_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == h // (2**i)
        outs[i].shape[3] == w // (2**i)

    bn_exist = False
    for m in mta_model.modules():
        if isinstance(m, _BatchNorm):
            bn_exist = True
    assert bn_exist

    # Extra convs source is 'inputs'
    mta_model = MTA(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_input',
        start_level=1,
        num_outs=5)
    assert mta_model.add_extra_convs == 'on_input'
    outs = mta_model(feats)
    assert len(outs) == mta_model.num_outs
    for i in range(mta_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == h // (2**i)
        outs[i].shape[3] == w // (2**i)

    # Extra convs source is 'outputs'
    mta_model = MTA(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_output',
        start_level=1,
        num_outs=5)
    assert mta_model.add_extra_convs == 'on_output'
    outs = mta_model(feats)
    assert len(outs) == mta_model.num_outs
    for i in range(mta_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == h // (2**i)
        outs[i].shape[3] == w // (2**i)
